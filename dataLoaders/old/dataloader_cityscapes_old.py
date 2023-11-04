from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, os.path
import torch
from PIL import *
from cityscapesscripts.helpers.labels import labels as city_labels
from torchvision.datasets import Cityscapes
import glob
import matplotlib.pyplot as plt
from torchvision import transforms




class CityscapesLoader(Dataset):


    def __init__(self, dataset_path, transform_in = None, transform_ou = None, mode = 'train'):

        super().__init__()

        self.dataset = mylist = [f for f in glob.glob(os.path.join(dataset_path, "images", mode, "*"))]
        self.city_labels = city_labels
        self.pixel_to_color = np.vectorize(self.return_color)

        self.transform_in = transform_in
        self.transform_ou = transform_ou

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.dataset[idx])
        seg_mask = np.array(Image.open(self.dataset[idx].replace("images", "masks")))
        seg_color = np.array(Image.open(self.dataset[idx].replace("images", "color")).convert('RGB'))
        #label = torch.tensor(self.create_prob_mask(seg_mask))
        label = self.create_prob_mask(seg_mask)

        if self.transform_in:
            img = self.transform_in(img)
            seg_color = transforms.ToTensor()(seg_color)
        if self.transform_ou:
            label = self.transform_ou(label)
            #seg_color = transforms.ToTensor()(self.transform_ou(seg_color))

        return {'image': img.float(), 'label': label, "seg": seg_color.float()}


    def get_num_classes(self):
        return len(self.city_labels)-1

    def create_prob_mask(self, seg_mask):

        seg_mask[seg_mask == -1] = 0
        seg_mask[seg_mask > 33] = 0
        label = np.zeros((seg_mask.shape[0], seg_mask.shape[1], self.get_num_classes()))
        indexs = np.ix_(np.arange(seg_mask.shape[0]), np.arange(seg_mask.shape[1]))
        label[indexs[0], indexs[1], seg_mask] = 1

        return label



    def return_color(self, idx):
        return self.city_labels[idx].color


    def prMask_to_color(self, img):

        argmax = torch.argmax(img, dim = 1)
        resu = self.pixel_to_color(argmax)
        return torch.tensor(np.transpose(np.stack((resu[0],resu[1], resu[2])), (1,0,2,3))).float()
