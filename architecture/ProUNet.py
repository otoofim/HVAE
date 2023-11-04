import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.distributions import Normal, Independent, kl
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl, MultivariateNormal
from torchmetrics.functional.classification import multiclass_calibration_error
from Prior import *
from Posterior import *
import sys
sys.path.insert(2, '../dataLoaders')
from MapillaryIntendedObjs import *
from sklearn.preprocessing import normalize


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.normal_(m.bias, std=0.001)



    
def multiclass_iou(output, target, classes):
    
    smooth = 1e-6
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    iou_scores = {}
    for cls, className in enumerate(classes):
        
        output_cls = torch.where(output == cls, 1, 0)
        target_cls = torch.where(target == cls, 1, 0)

        intersection = (output_cls * target_cls).sum()
        union = (output_cls + target_cls).sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_scores[className] = iou.nanmean()
        
        
    mean_iou = torch.stack(list(iou_scores.values())).nanmean()
    return mean_iou, iou_scores
    
def SingleImageConfusionMatrix(Predictions, Lables, classes):
    
#     smooth = 1e-6
    CM_unnormalised = torch.zeros([8,8]) 
    total_pixels_in_class_j = torch.zeros([8,1])
    Predictions = torch.argmax(Predictions, dim=0)
    Lables = torch.argmax(Lables, dim=0)

    # produce unnormalised CM:
    for j,_ in enumerate(classes):
        is_pixel_in_class_j = torch.where(Lables == j, 1, 0)
        total_pixels_in_class_j[j] = is_pixel_in_class_j.sum()

        for i,_ in enumerate(classes):

            is_pixel_predicted_as_i = torch.where(Predictions == i, 1, 0)
            is_pixel_predicted_as_i_and_is_in_class_j = is_pixel_in_class_j * is_pixel_predicted_as_i
            total_pixels_predicted_as_i_in_class_j = is_pixel_predicted_as_i_and_is_in_class_j.sum()
            CM_unnormalised[i][j] = total_pixels_predicted_as_i_in_class_j

    return np.array([CM_unnormalised, total_pixels_in_class_j])


def BatchImageConfusionMatrix(predictions, labels, classes):
    
    smooth = 1e-3
    smooth = 0.
    pixels_in_class_j_totals = torch.zeros([8,1])
    CM_unnormalised_totals = torch.zeros([8,8])

    confusionTest = [SingleImageConfusionMatrix(pred, sample, classes) for pred, sample in zip(predictions, labels)]
    
    CM_unnormalised_totals = np.nan_to_num(np.array(list(zip(*confusionTest))[0])).sum()
    pixels_in_class_j_totals = np.nan_to_num(np.array(list(zip(*confusionTest))[1])).sum()
                                             
                                             
                                             
                                             
    CM = torch.tensor([100 * ((CM_unnormalised_totals[i][j] + smooth) / (pixels_in_class_j_totals[j] + smooth)) for i,_ in enumerate(classes) for j,_ in enumerate(classes)]).reshape(8,8)
    
    return CM
    
    
class MyMSE(torch.nn.Module):
    def __init__(self):
        super(MyMSE, self).__init__()

        
    def forward(self, output, target):
        
        mask = target.detach() != 0
        loss = torch.pow(target-output, 2)
        loss = loss*mask
        loss_mean = loss.sum()/(mask.sum()+1e-15)

        return loss_mean  

            


        
        
class CrossEntopy(torch.nn.Module):
    def __init__(self, label_smoothing):
        super(CrossEntopy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing = label_smoothing)

    
    def forward(self, output, target):

        CEL = torch.mean(self.criterion(input = output, target = target),(1,2))
        return CEL
    
    
    
class GECO():
    
# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2021 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================
#https://github.com/applied-ai-lab/genesis/blob/master/utils/geco.py
    def __init__(self, goal, step_size, device, alpha=0.99, beta_init=1.,
                 beta_min=1e-10, speedup=None):
        self.err_ema = None
        self.goal = goal
        self.step_size = step_size
        self.alpha = alpha
        self.beta = torch.tensor(beta_init)
        self.beta_min = torch.tensor(beta_min)
        self.beta_max = torch.tensor(1e10)
        self.speedup = speedup
        self.device = device
        self.to_device()

    def to_device(self):
        self.beta = self.beta.to(device=self.device)
        self.beta_min = self.beta_min.to(device=self.device)
        self.beta_max = self.beta_max.to(device=self.device)
        if self.err_ema is not None:
            self.err_ema = self.err_ema.to(device=self.device)

    def loss(self, err, kld):
        # Compute loss with current beta
        loss = err + self.beta * kld
        # Update beta without computing / backpropping gradients
        with torch.no_grad():
            if self.err_ema is None:
                self.err_ema = err
            else:
                self.err_ema = (1.0-self.alpha)*err + self.alpha*self.err_ema
            constraint = (self.goal - self.err_ema)
            if self.speedup is not None and constraint.item() > 0:
                factor = torch.exp(self.speedup * self.step_size * constraint)
            else:
                factor = torch.exp(self.step_size * constraint)
            self.beta = (factor * self.beta).clamp(self.beta_min, self.beta_max)
        # Return loss
        return loss
##################################################################################################



        
class ProUNet(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, num_classes, gecoConfig, device, LatentVarSize = 6, beta = 5., training = True, num_samples = 16):
        super(ProUNet, self).__init__()
        #Vars init
        self.LatentVarSize = LatentVarSize
        self.beta = beta
        self.training = training
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.gecoConfig = gecoConfig
        self.device = device
        
        if self.gecoConfig["enable"]:
            self.geco = GECO(goal = gecoConfig["goal"], alpha = gecoConfig["alpha"], speedup = gecoConfig["speedup"], beta_init = gecoConfig["beta_init"], step_size = gecoConfig["step_size"], device = self.device)


        #architecture
        self.prior = prior(self.num_samples, self.num_classes, self.LatentVarSize).apply(init_weights)
        if training:
            self.posterior = posterior(self.num_samples, self.num_classes, self.LatentVarSize).apply(init_weights)
        
        #loss functions
        self.criterion = CrossEntopy(label_smoothing = 0.4)
        self.regressionLoss = MyMSE()

        
    def forward(self, inputImg, segmasks = None, friLabel = None):
        
        posteriorDists = self.posterior(torch.cat((inputImg, segmasks, friLabel), 1))
        seg, priorDists, fri = self.prior(inputImg, postDist = posteriorDists)
    
        return seg, priorDists, posteriorDists, fri
        
        
    def inference(self, inputFeatures):
        with torch.no_grad():
            return self.prior.inference(inputFeatures)
    
    
    def evaluation(self, inputFeatures, segmasks, friLabel):
        
        with torch.no_grad():
            samples, priors, fris = self.prior.inference(inputFeatures)
            posteriorDists = self.posterior.inference(torch.cat((inputFeatures, segmasks, friLabel), 1))
            return samples, priors, posteriorDists, fris
    
    
    def rec_loss(self, img, seg):

        error = self.criterion(output = img, target = seg)
        return error

    
    def kl_loss(self, priors, posteriors):
        
        klLoss = {}
        for level, (posterior, prior) in enumerate(zip(posteriors.items(), priors.items())):
            klLoss[level] = torch.mean(kl.kl_divergence(posterior[1], prior[1]), (1,2))
        return klLoss
    
    
    def elbo_loss(self, label, seg, priors, posteriors, friLabel = None, friPred = None):
        
        rec_loss = torch.mean(self.rec_loss(label, seg))
        
        kl_losses = self.kl_loss(priors, posteriors)
        kl_mean = torch.mean(torch.sum(torch.stack([i for i in kl_losses.values()]), 0))
        
        regLoss = self.regressionLoss(target = friLabel, output = friPred)
        
        loss = torch.mean(rec_loss + (self.beta * kl_mean) + regLoss)
#         loss = torch.mean(rec_loss + self.beta * kl_mean)
        
        return loss, kl_mean, kl_losses, rec_loss, regLoss

    
    
    def stats(self, predictions, labels):
        
        
        miou, ious = multiclass_iou(predictions, labels, classIds)
        CM = BatchImageConfusionMatrix(predictions, labels, classIds)
        
    
        l1Loss = multiclass_calibration_error(preds = predictions, target = torch.argmax(labels, 1), num_classes = self.num_classes, n_bins = 10, norm = 'l1')
        l2Loss = multiclass_calibration_error(preds = predictions, target = torch.argmax(labels, 1), num_classes = self.num_classes, n_bins = 10, norm = 'l2')
        l3Loss = multiclass_calibration_error(preds = predictions, target = torch.argmax(labels, 1), num_classes = self.num_classes, n_bins = 10, norm = 'max')
    
        return miou, ious, l1Loss, l2Loss, l3Loss, CM
    
    def lossGECO(self, label, segPred, priors, posteriors, friLabel = None, friPred = None):
        
        rec_loss = torch.mean(self.rec_loss(label, segPred))
        
        kl_losses = self.kl_loss(priors, posteriors)
        kl_mean = torch.mean(torch.sum(torch.stack([i for i in kl_losses.values()]), 0))
        
        regLoss = self.regressionLoss(target = friLabel, output = friPred)
 
        loss = self.geco.loss(rec_loss + regLoss, kl_mean)
    
        return loss, kl_mean, kl_losses, rec_loss, regLoss
    

    def loss(self, label, segPred, priors, posteriors, friLabel = None, friPred = None):
        
        
        if self.gecoConfig["enable"]:
            loss, kl_mean, kl_losses, rec_loss, regLoss = self.lossGECO(label, segPred, priors, posteriors, friLabel, friPred)
        else:
            loss, kl_mean, kl_losses, rec_loss, regLoss = self.elbo_loss(label, seg, priors, posteriors, friLabel, friPred)

            
        miou, ious, l1Loss, l2Loss, l3Loss, CM = self.stats(segPred, label)
        
        return loss, kl_mean, kl_losses, rec_loss, miou, ious, l1Loss, l2Loss, l3Loss, regLoss, CM

    
        
        
