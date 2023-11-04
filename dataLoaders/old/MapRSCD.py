from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, os.path
import torch
from pathlib import Path
from os.path import exists
#from PIL import *
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from cityscapesscripts.helpers.labels import labels as city_labels
from torchvision.datasets import Cityscapes
import glob
from torchvision import transforms
from matplotlib import cm
import sys
import json
import torchvision
from MapillaryIntendedObjs import *
from augmentation import *




class MapRSCD(Dataset):


    def __init__(self, datasetRootPath, RSCDAdd = "../../datasets/RSCD", mapillaryConfigFileAdd = "../../../datasets/mapillary", mode = 'train', imgSize = (256,256)):

        super().__init__()

        if mode not in ["train", "val", "test"]:
            raise ValueError("valid values for mode arguement are: train, val, test")
        
        self.mode = mode
        self.imgSize = imgSize
        self.RSCD_cat = ["ice", "fresh_snow"]
        datasetRootPath = Path(datasetRootPath)
        
        tmpDataset = []
        for imgPath in glob.glob(str(Path.joinpath(datasetRootPath, "images", mode, "*.png"))):
            
            if (exists(str(imgPath).replace("images","color"))) and (exists(str(imgPath).replace("images","masks"))):
                tmpDataset.append(imgPath)
        
        if mode == "val":
            tmpDataset.extend([path for path in glob.glob(str(Path.joinpath(Path(RSCDAdd), mode, "*.jpg"))) if any([cat in path.split(os.sep)[-1].split(".")[0] for cat in self.RSCD_cat])])
        elif mode == "train":
            for cat in self.RSCD_cat:
                tmpDataset.extend(glob.glob(str(Path.joinpath(Path(RSCDAdd), mode, cat, "*.jpg"))))
        
        self.dataset = np.array(tmpDataset)  
        
        self.newLabels = new_labels
        self.classIds = classIds
        self.RSCDClassNames = RSCDClassNames
        
        
        labels = {}
        add = Path(mapillaryConfigFileAdd)
        with open(str(Path.joinpath(add, "config_v2.0.json"))) as jsonfile:
            config = json.load(jsonfile)
            for label in config['labels']:
                labels[label['name']] = label['color']
        self.labels = labels
        
        
        
        self.pixel_to_color = np.vectorize(self.return_color)


        
        self.transform_in = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4191, 0.4586, 0.4700], [0.2553, 0.2675, 0.2945]),
            transforms.Resize(imgSize)
        ])
        
        self.transform_ou = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(imgSize)
        ])

    def __len__(self):
        return len(self.dataset)

    def get_num_classes(self):
        return len(self.classIds)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        img = Image.open(self.dataset[idx])
        if "RSCD" in self.dataset[idx]:
            cat = self.RSCDClassNames[self.dataset[idx].split(os.sep)[-1].split(".")[0].split("-")[1]]
            seg_mask = np.full_like(np.array(img), list(self.classIds.keys()).index(cat))
            label, seg_color = self.create_prob_mask_patches(seg_mask[:,:,0])
            
            
        else:
            seg_mask = np.array(Image.open(self.dataset[idx].replace("images", "masks")))
            seg_color = np.array(Image.open(self.dataset[idx].replace("images", "color")).convert('RGB'))
            label, seg_color = self.create_prob_mask(seg_mask, seg_color)
            
            #seg_color = seg_color[:,:,0]

        if self.transform_in:
            img = self.transform_in(img)
            seg_color = transforms.Resize((256,256))(transforms.ToTensor()(seg_color))
        if self.transform_ou:
            label = self.transform_ou(label)
            #labels = torch.stack([self.transform_ou(sample) for sample in labels])

        return {'image': img.type(torch.float), 'label': label.type(torch.float), "seg": seg_color.type(torch.float)}


    def create_prob_mask(self, seg_mask, seg_color):
        
        masks = np.stack([seg_mask for _ in range(len(self.classIds))], axis=0)
        
        for i, label in enumerate(self.labels):
            classid = list(self.classIds.keys())[list(self.classIds.values()).index(self.newLabels[label])]
            seg_color[np.all(seg_color == self.labels[label], axis=-1)] = self.classIds[classid]
            seg_mask[seg_mask == i] = list(self.classIds.keys()).index(classid)

        label = np.zeros((seg_mask.shape[0], seg_mask.shape[1], len(self.classIds)))
        indexs = np.ix_(np.arange(seg_mask.shape[0]), np.arange(seg_mask.shape[1]))
        label[indexs[0], indexs[1], seg_mask] = 1
        #labels = self.seperateClasses(label)

        #return label, seg_color, labels
        return label, seg_color
    
    
    def create_prob_mask_patches(self, seg_mask):

        label = np.zeros((seg_mask.shape[0], seg_mask.shape[1], len(self.classIds)))
        indexs = np.ix_(np.arange(seg_mask.shape[0]), np.arange(seg_mask.shape[1]))
        label[indexs[0], indexs[1], seg_mask] = 1
        seg_color = self.prMask_to_color(torch.tensor(label).permute(2,0,1).unsqueeze(0))
        seg_color = seg_color.squeeze(0).permute(1,2,0).cpu().detach().numpy()

        #return label, labels
        return label, seg_color
    
    def return_color(self, idx):
        return tuple(self.classIds[list(self.classIds.keys())[int(idx)]])

    def prMask_to_color(self, img):
        argmax = torch.argmax(img, dim = 1)
        resu = self.pixel_to_color(argmax)
        return (torch.tensor(np.transpose(np.stack((resu[0],resu[1], resu[2])), (1,0,2,3))).float())/255

    
    
    def seperateClasses(self, seg_mask):
        
        segMasks = []
        for classLabel in range(len(self.classIds)):
             
            argmaxs = torch.argmax(torch.tensor(seg_mask), axis = -1)
            converted = torch.where(argmaxs == classLabel, argmaxs, 0)

            label = np.zeros((converted.shape[0], converted.shape[1], len(self.classIds)))
            indexs = np.ix_(np.arange(converted.shape[0]), np.arange(converted.shape[1]))
            label[indexs[0], indexs[1], converted] = 1
            segMasks.append(label)
            
        return np.array(segMasks)
        
        
        
        