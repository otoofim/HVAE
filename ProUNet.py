import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.distributions import Normal, Independent, kl
import torchvision
import torchvision.transforms as T
from torch.distributions import Normal, Independent, kl, MultivariateNormal
from UNet import *

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(m.weight, std=0.001)
        nn.init.normal_(m.bias, std=0.001)

        #truncated_normal_(m.bias, mean=0, std=0.001)    
    
class ProUNet(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, num_classes, LatentVarSize = 6, beta = 5., training = True, num_samples = 16):
        super(ProUNet, self).__init__()
        #Vars init
        self.LatentVarSize = LatentVarSize
        self.beta = beta
        self.training = training
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        #architecture
        self.unet = UNet(self.num_samples, self.num_classes, self.LatentVarSize).apply(init_weights)
        if training:
            self.posterior = UNet(self.num_samples, self.num_classes, self.LatentVarSize, posterior = True).apply(init_weights)
        
        #loss functions
        self.criterion = nn.BCEWithLogitsLoss(size_average = False, reduction = None, reduce = False)
        
    def forward(self, inputImg, segmasks = None):
        
        seg, priorDists = self.unet(inputImg)
        
        if self.training:
            posteriorDists = self.posterior(torch.cat((inputImg, segmasks), 1))
            return seg, priorDists, posteriorDists
        
        return seg, priorDists
        
    def inference(self, inputFeatures):
        with torch.no_grad():
            return self.unet.inference(inputFeatures)
    
    
    def evaluation(self, inputFeatures, segmasks):
        
        with torch.no_grad():
            samples, priors = self.unet.inference(inputFeatures)
            posteriorDists = self.posterior(torch.cat((inputFeatures, segmasks), 1))
            return samples, priors, posteriorDists
    
    
    def rec_loss(self, img, seg):
        return self.criterion(input = img, target = seg)
    
    
    def kl_loss(self, priors, posteriors):
        
        klLoss = {}
        for level, (posterior, prior) in enumerate(zip(posteriors.items(), priors.items())):
            klLoss[level] = torch.mean(kl.kl_divergence(posterior[1], prior[1]), (1,2))
        return klLoss
    
    
    def elbo_loss(self, label, seg, priors, posteriors):
        
        rec_loss = torch.sum(torch.mean(self.rec_loss(label, seg),(2,3)),1)
        kl_losses = self.kl_loss(priors, posteriors)
        kl_mean = torch.mean(torch.stack([i for i in kl_losses.values()]), 0)
        
        loss = torch.mean(rec_loss + self.beta * kl_mean)
        
        return loss, torch.mean(kl_mean), kl_losses, torch.mean(rec_loss)

