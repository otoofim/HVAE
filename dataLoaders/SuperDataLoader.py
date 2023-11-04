from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, os.path
import torch
from pathlib import Path
from os.path import exists
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from cityscapesscripts.helpers.labels import labels as city_labels
from torchvision.datasets import Cityscapes
import glob
import json
from os.path import join
from torchvision import transforms
from MapillaryIntendedObjs import *




class SuperDataLoader(Dataset):
    
    def __init__(self, **kwargs):
        
        super().__init__()
        if kwargs["mode"] not in ["train", "val"]:
            raise ValueError("valid values for mode arguement are: train, val, test")
        
                
        self.mode = kwargs["mode"]
        self.imgSize = kwargs["input_img_dim"]
        self.reducedCategoriesColors = classIds
        self.friClass = friClass
        self.reducedCategories = kwargs["reducedCategories"] 
    
        self.pixel_to_color = np.vectorize(self.return_color)
    
    
    def create_prob_mask(self, seg_mask, seg_color):
        
        fricLabel = torch.zeros(seg_mask.shape)
        
        if self.reducedCategories:
            
            tmpSeg = np.zeros(seg_mask.shape)
            for i, label in enumerate(self.labels):
                classid = list(self.reducedCategoriesColors.keys())[list(self.reducedCategoriesColors.values()).index(self.NewColors[label])]
                seg_color[np.all(seg_color == self.labels[label], axis=-1)] = self.reducedCategoriesColors[classid]
                tmpSeg[seg_mask == i] = list(self.reducedCategoriesColors.keys()).index(classid)
                
            seg_mask = tmpSeg.astype(int)
            #creating friction label
            for i,className in enumerate(self.reducedCategoriesColors):
                fricLabel = np.where(seg_mask == i, self.friClass[className], fricLabel)
        
        label = np.zeros((seg_mask.shape[0], seg_mask.shape[1], self.get_num_classes()))
        indexs = np.ix_(np.arange(seg_mask.shape[0]), np.arange(seg_mask.shape[1]))
        label[indexs[0], indexs[1], seg_mask] = 1
        return label, seg_color, fricLabel
    
    
    def create_prob_mask_patches(self, seg_mask):

        label = np.zeros((seg_mask.shape[0], seg_mask.shape[1], self.get_num_classes()))
        indexs = np.ix_(np.arange(seg_mask.shape[0]), np.arange(seg_mask.shape[1]))
        label[indexs[0], indexs[1], seg_mask] = 1
        fricLabel = np.ones(seg_mask.shape)
#         print(fricLabel.shape)
        fricLabel *= list(self.friClass.values())[seg_mask[0,0]]
        seg_color = self.prMask_to_color(torch.tensor(label).permute(2,0,1).unsqueeze(0))
        seg_color = seg_color.squeeze(0).permute(1,2,0).cpu().detach().numpy()
        return label, seg_color, fricLabel
    
    def return_color(self, idx):
        if self.reducedCategories:
            return tuple(self.reducedCategoriesColors[list(self.reducedCategoriesColors.keys())[int(idx)]])
        else:
            return tuple(self.labels[list(self.labels.keys())[int(idx)]])
    
    def prMask_to_color(self, img):
        argmax = torch.argmax(img, dim = 1)
        resu = self.pixel_to_color(argmax)
        return torch.tensor(np.transpose(np.stack((resu[0],resu[1], resu[2])), (1,0,2,3))).float()/255
    
    def seperateClasses(self, seg_mask):
        
        segMasks = []
        for classLabel in range(len(self.reducedCategoriesColors)):
             
            argmaxs = torch.argmax(torch.tensor(seg_mask), axis = -1)
            converted = torch.where(argmaxs == classLabel, argmaxs, 0)

            label = np.zeros((converted.shape[0], converted.shape[1], len(self.reducedCategoriesColors)))
            indexs = np.ix_(np.arange(converted.shape[0]), np.arange(converted.shape[1]))
            label[indexs[0], indexs[1], converted] = 1
            segMasks.append(label)
            
        return np.array(segMasks)
    
    