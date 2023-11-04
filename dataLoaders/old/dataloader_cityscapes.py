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
import sys
import json
import torchvision
#sys.path.insert(1, '../mapillary')
from MapillaryIntendedObjs import *
from augmentation import *




class CityscapesLoader(Dataset):


    def __init__(self, datasetRootPath, mode = 'train', imgSize = (256,256)):

        super().__init__()

        if mode not in ["train", "val", "test"]:
            raise ValueError("valid values for mode arguement are: train, val, test")
        
        self.mode = mode
        self.imgSize = imgSize
        datasetRootPath = Path(datasetRootPath)
        
        tmpDataset = []
        for imgPath in glob.glob(str(Path.joinpath(datasetRootPath, "images", mode, "*.jpg"))):
            
            if (exists(str(imgPath).replace("images","color"))) and (exists(str(imgPath).replace("images","masks"))):
                tmpDataset.append(imgPath)
        
        self.dataset = np.array(tmpDataset)  
        
        self.newLabels = new_labels
        self.classIds = classIds
        self.labels = {label.name:[label.id, label.color] for label in city_labels}
        self.labels["bicyclev"] = self.labels["bicycle"]
        del self.labels["bicycle"]
        self.pixel_to_color = np.vectorize(self.return_color)



        self.transform_in = preprocess_in = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.2867, 0.3250, 0.2837], [0.1862, 0.1895, 0.1865]),
    transforms.Resize(imgSize)
])
        self.transform_ou = preprocess_out = transforms.Compose([
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
        seg_mask = np.array(Image.open(self.dataset[idx].replace("images", "masks")))
        seg_color = np.array(Image.open(self.dataset[idx].replace("images", "color")).convert('RGB'))
        label, seg_color = self.create_prob_mask(seg_mask, seg_color)

        if self.transform_in:
            img = self.transform_in(img)
            seg_color = transforms.ToTensor()(seg_color)
        if self.transform_ou:
            label = self.transform_ou(label)

        return {'image': img, 'label': label, "seg": seg_color}


    def create_prob_mask(self, seg_mask, seg_color):
        
        mask = seg_mask.copy()
        listOfKeys = [kk.split("--")[-1] for kk in list(self.newLabels.keys())]
        for label in self.labels:
            tmpKey = list(self.newLabels.keys())[listOfKeys.index(label.replace(" ", "-"))]
            classid = list(self.classIds.keys())[list(self.classIds.values()).index(self.newLabels[tmpKey])]
            #print(tmpKey)
            seg_mask[seg_mask == self.labels[label][0]] = list(self.classIds.keys()).index(classid)
            seg_color[np.all(seg_color == self.labels[label][1], axis=-1)] = self.classIds[classid]

        label = np.zeros((seg_mask.shape[0], seg_mask.shape[1], self.get_num_classes()))
        indexs = np.ix_(np.arange(seg_mask.shape[0]), np.arange(seg_mask.shape[1]))
        label[indexs[0], indexs[1], seg_mask] = 1

        return label, seg_color
    

    def return_color(self, idx):
        return self.city_labels[idx].color
        return tuple(classIds[list(classIds.keys())[int(idx)]])


    def prMask_to_color(self, img):
        
        argmax = torch.argmax(img, dim = 1)
        resu = self.pixel_to_color(argmax)
        return (torch.tensor(np.transpose(np.stack((resu[0],resu[1], resu[2])), (1,0,2,3))).float())/255
