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
sys.path.insert(1, '../mapillary')
from MapillaryIntendedObjs import *
from augmentation import *



class CityscapesLoader(Dataset):


    def __init__(self, datasetRootPath = "../datasets/Cityscapes", mode = 'train', imgSize = (256,256)):

        super().__init__()
        
        if mode not in ["train", "val", "test"]:
            raise ValueError("valid values for mode arguement are: train, val, test")
        
        self.mode = mode
        self.imgSize = imgSize
        datasetRootPath = Path(datasetRootPath)
        
        tmpDataset = []
        for imgPath in glob.glob(str(Path.joinpath(datasetRootPath, "leftImg8bit", mode, "*", "*.png"))):
            
            if (exists(str(imgPath).replace("leftImg8bit","gtFine").replace(".png", "_color.png"))) and (exists(str(imgPath).replace("leftImg8bit","gtFine").replace(".png", "_labelIds.png"))):
                tmpDataset.append(imgPath)
        
        listDataset = []
        for add in tmpDataset:
            listDataset += [add+"***"+str(j) for j in range(11)]
              
        self.dataset = np.array(listDataset)  

        self.newLabels = new_labels
        self.classIds = classIds
        self.labels = {label.name:[label.id, label.color] for label in city_labels}
        self.labels["bicyclev"] = self.labels["bicycle"]
        del self.labels["bicycle"]
        self.pixel_to_color = np.vectorize(self.return_color)


    def __len__(self):
        return len(self.dataset)

    def get_num_classes(self):
        return len(self.classIds)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        add = self.dataset[idx].split("***")[0]
        transformid = int(self.dataset[idx].split("***")[1])
        img = Image.open(add).convert('RGB')
        segColorAdd = str(add).replace("leftImg8bit","gtFine").replace(".png", "_color.png")
        labelAdd = str(add).replace("leftImg8bit","gtFine").replace(".png", "_labelIds.png")
        seg_color = Image.open(segColorAdd).convert('RGB')
        seg_label = Image.open(labelAdd).convert('RGB')
                
        img, seg_color, seg_label = self.myAugmentation(transformid, img, seg_color, seg_label)
        
        
        #It turns seg mask to the intended classes and return prob mask
        seg_label, seg_color = self.create_prob_mask(seg_label, seg_color)
        
        seg_color = transforms.ToTensor()(seg_color)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.2867, 0.3250, 0.2837], [0.1862, 0.1895, 0.1865])(img)
        seg_label = transforms.ToTensor()(seg_label)


        return {'image': img, 'label': seg_label, "seg": seg_color}


    
    def create_prob_mask(self, seg_mask, seg_color):
        
        mask = seg_mask.copy()
        for label in self.labels:
            listOfKeys = [kk.split("--")[-1] for kk in list(self.newLabels.keys())]
            tmpKey = list(self.newLabels.keys())[listOfKeys.index(label.replace(" ", "-"))]
            
            classid = list(self.classIds.keys())[list(self.classIds.values()).index(self.newLabels[tmpKey])]
            seg_mask[np.all(seg_mask == self.labels[label][0], axis=-1)] = list(self.classIds.keys()).index(classid)
            seg_color[np.all(seg_color == self.labels[label][1], axis=-1)] = self.classIds[classid]

        label = np.zeros((seg_mask.shape[0], seg_mask.shape[1], self.get_num_classes()))
        indexs = np.ix_(np.arange(seg_mask.shape[0]), np.arange(seg_mask.shape[1]))
        label[indexs[0], indexs[1], seg_mask[:,:,0]] = 1

        return label, seg_color
    
    
    def myAugmentation(self, transformid, img, seg_color, seg_label):
        
        final_image_dim = self.imgSize
        input_image_shape = np.array(img).shape
        #These normalization values are calculated based on the whole cityscapes dataset
        #values from ImageNet mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        #img = transforms.Normalize([0.2867, 0.3250, 0.2837], [0.1862, 0.1895, 0.1865])(transforms.ToTensor()(img))
        
        if transformid == 10:
            img = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(img)
            seg_color = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(seg_color)
            seg_label = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(seg_label)
            
        elif transformid%2 == 0:
            img = transforms.FiveCrop((int(input_image_shape[0]/2), int(input_image_shape[1]/2)))(img)
            img = img[int(transformid/2)]
            img = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(img)
            
            seg_color = transforms.FiveCrop((int(input_image_shape[0]/2), int(input_image_shape[1]/2)))(seg_color)
            seg_color = seg_color[int(transformid/2)]
            seg_color = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(seg_color)
            
            seg_label = transforms.FiveCrop((int(input_image_shape[0]/2), int(input_image_shape[1]/2)))(seg_label)
            seg_label = seg_label[int(transformid/2)]
            seg_label = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(seg_label)

        else:
            img = transforms.FiveCrop((int(input_image_shape[0]/2), int(input_image_shape[1]/2)))(img)
            img = img[int((transformid-1)/2)]
            transf = random.choice(customized_augment_transforms)
            img = transf(img)
            img = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(img)
            
            seg_color = transforms.FiveCrop((int(input_image_shape[0]/2), int(input_image_shape[1]/2)))(seg_color)
            seg_color = seg_color[int((transformid-1)/2)]
            
            seg_label = transforms.FiveCrop((int(input_image_shape[0]/2), int(input_image_shape[1]/2)))(seg_label)
            seg_label = seg_label[int((transformid-1)/2)]
            
            if isinstance(transf, RandomAffine):
                seg_color = transf(seg_color)
                seg_color = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(seg_color)
                
                seg_label = transf(seg_label)
                seg_label = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(seg_label)
                
            else:
                seg_color = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(seg_color)
                seg_label = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(seg_label)

        return np.array(img), np.array(seg_color), np.array(seg_label)
    
    
    def return_color(self, idx):
        return tuple(classIds[list(classIds.keys())[int(idx)]])


    def prMask_to_color(self, img):
        
        argmax = torch.argmax(img, dim = 1)
        resu = self.pixel_to_color(argmax)
        return (torch.tensor(np.transpose(np.stack((resu[0],resu[1], resu[2])), (1,0,2,3))).float())/255
    
    
    
    
    
               