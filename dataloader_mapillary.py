from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, os.path
import torch
from pathlib import Path
#from PIL import *
import PIL.Image as Image
import glob
from torchvision import transforms
import sys
import json
import torchvision
sys.path.insert(1, '../mapillary')
from MapillaryIntendedObjs import *
from augmentation import *



class MapillaryLoader(Dataset):


    def __init__(self, datasetRootPath = "../mapillary", mode = 'training', ver = "v2.0", imgSize = (256,256)):

        super().__init__()
        
        self.mode = mode
        self.ver = ver
        self.imgSize = imgSize
        datasetRootPath = Path(datasetRootPath)
        tmpDataset = [f for f in glob.glob(str(Path.joinpath(datasetRootPath, mode, "images", "*.jpg")))][:100]
        self.dataset = np.array([])
        for add in tmpDataset:
            self.dataset = np.append(self.dataset, [add+"***"+str(j) for j in range(11)])
              
        self.newLabels = new_labels
        self.classIds = classIds
        self.labels = {}
        with open(str(Path.joinpath(datasetRootPath, "config_{}.json".format(self.ver)))) as jsonfile:
            config = json.load(jsonfile)
            for label in config['labels']:
                self.labels[label['name']] = label['color']

        self.transform_in = transform_in
        self.transform_ou = transform_ou

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
        labelAdd = str(add).replace("images", self.ver + os.sep + "labels").replace("jpg", "png")
        seg_color = Image.open(labelAdd).convert('RGB')
        img, seg_color = self.myAugmentation(transformid, img, seg_color)
        
        
        #It turns seg mask to the intended classes and return prob mask
        probMask, seg_colorNew = self.create_prob_mask(seg_color)
        seg_colorNew = transforms.ToTensor()(seg_colorNew)
        img = transforms.ToTensor()(img)
        probMask = transforms.ToTensor()(probMask)


        return {'image': img, 'label': probMask, "seg": seg_colorNew}


    
    def create_prob_mask(self, seg_mask):

        mask = seg_mask.copy()
        for label in self.labels:
            classid = list(self.classIds.keys())[list(self.classIds.values()).index(self.newLabels[label])]
            seg_mask[np.all(seg_mask == self.labels[label], axis=-1)] = list(self.classIds.keys()).index(classid)
            mask[np.all(mask == self.labels[label], axis=-1)] = self.newLabels[label]

        label = np.zeros((seg_mask.shape[0], seg_mask.shape[1], self.get_num_classes()))
        indexs = np.ix_(np.arange(seg_mask.shape[0]), np.arange(seg_mask.shape[1]))
        label[indexs[0], indexs[1], seg_mask[:,:,0]] = 1

        return label, mask
    
    
    def myAugmentation(self, transformid, img, seg_color):
        
        final_image_dim = self.imgSize
        input_image_shape = np.array(img).shape
        
        if transformid == 10:
            img = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(img)
            seg_color = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(seg_color)
            
        elif transformid%2 == 0:
            img = transforms.FiveCrop((int(input_image_shape[0]/2), int(input_image_shape[1]/2)))(img)
            img = img[int(transformid/2)]
            img = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(img)
            
            seg_color = transforms.FiveCrop((int(input_image_shape[0]/2), int(input_image_shape[1]/2)))(seg_color)
            seg_color = seg_color[int(transformid/2)]
            seg_color = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(seg_color)

        else:
            img = transforms.FiveCrop((int(input_image_shape[0]/2), int(input_image_shape[1]/2)))(img)
            img = img[int((transformid-1)/2)]
            transf = random.choice(customized_augment_transforms)
            img = transf(img)
            img = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(img)
            
            seg_color = transforms.FiveCrop((int(input_image_shape[0]/2), int(input_image_shape[1]/2)))(seg_color)
            seg_color = seg_color[int((transformid-1)/2)]
            
            if isinstance(transf, RandomAffine):
                seg_color = transf(seg_color)
                seg_color = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(seg_color)
            else:
                seg_color = transforms.Resize(final_image_dim, torchvision.transforms.InterpolationMode.NEAREST)(seg_color)

        return np.array(img), np.array(seg_color)
               
        
    
