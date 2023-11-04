from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, os.path
import torch
from pathlib import Path
#from PIL import *
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import glob
from torchvision import transforms
import sys
import json
import torchvision
#sys.path.insert(1, '../mapillary')
from MapillaryIntendedObjs import *
from augmentation import *



class MapillaryLoader(Dataset):


    def __init__(self, datasetRootPath = "../mapillary", mode = 'training', ver = "v2.0", imgSize = (256,256)):

        super().__init__()
        
        if mode not in ["training", "validation", "testing"]:
             raise ValueError("valid values for mode arguement are: training, validation, testing")
        if ver not in ["v2.0", "v1.2"]:
            raise ValueError("valid values for ver arguement are: v2.0, v1.2")
        
        self.mode = mode
        self.ver = ver
        self.imgSize = imgSize
        datasetRootPath = Path(datasetRootPath)
        tmpDataset = [f for f in glob.glob(str(Path.joinpath(datasetRootPath, mode, "images", "*.jpg")))]
        listDataset = []
        for add in tmpDataset:
            listDataset += [add+"***"+str(j) for j in range(11)]
              
        self.dataset = np.array(listDataset)        
        self.newLabels = new_labels
        self.classIds = classIds
        self.labels = {}
        with open(str(Path.joinpath(datasetRootPath, "config_{}.json".format(self.ver)))) as jsonfile:
            config = json.load(jsonfile)
            for label in config['labels']:
                self.labels[label['name']] = label['color']
        
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
        labelAdd = str(add).replace("images", self.ver + os.sep + "labels").replace("jpg", "png")
        seg_color = Image.open(labelAdd).convert('RGB')
        img, seg_color = self.myAugmentation(transformid, img, seg_color)
        
        
        #It turns seg mask to the intended classes and return prob mask
        probMask, seg_colorNew = self.create_prob_mask(seg_color)
        seg_colorNew = transforms.ToTensor()(seg_colorNew)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.4191, 0.4586, 0.4700], [0.2553, 0.2675, 0.2945])(img)
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
        #These normalization values are calculated based on the whole mapillary dataset
        #img = transforms.Normalize([0.4191, 0.4586, 0.4700], [0.2553, 0.2675, 0.2945])(img)
        
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
               
    def return_color(self, idx):
        return tuple(classIds[list(classIds.keys())[int(idx)]])


    def prMask_to_color(self, img):

        argmax = torch.argmax(img, dim = 1)
        resu = self.pixel_to_color(argmax)
        return (torch.tensor(np.transpose(np.stack((resu[0],resu[1], resu[2])), (1,0,2,3))).float())/255
    
