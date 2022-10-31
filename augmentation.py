import sys
import os
import glob
from PIL import Image
import numpy as np
from torchvision import transforms
import random
import torch
import torchvision.transforms.functional as TF


class RandomErasing:

    def __call__(self, x):

        if not isinstance(x, torch.Tensor):
            x = transforms.ToTensor()(x)
        x = transforms.ToPILImage()(transforms.RandomErasing(p=1.)(x))
        return x

class RandomAffine:

    def __init__(self, angles = (0,360), translate = (2,2), scale = (0,2), shear = (-180,180)):
        self.angle_range = angles
        self.translate = translate
        self.scale_range = scale
        self.shear_range = shear
        self.__new_seed__()

    def __call__(self, x):
        return TF.affine(x, self.angle, self.translate, self.scale, self.shear)

    def __new_seed__(self):
        self.angle = random.uniform(*self.angle_range)
        self.scale = random.uniform(*self.scale_range)
        self.shear = random.uniform(*self.shear_range)

        
customized_augment_transforms = [
    transforms.ColorJitter(brightness = (0.5,2), contrast = (0.5,2), saturation = (0.5,2), hue = (-0.5,0.5)),
    RandomErasing(),
    transforms.GaussianBlur(9),
    transforms.Grayscale(num_output_channels = 3),
    RandomAffine(angles = (0,360), translate = (0.5,2), scale = (0.5,2), shear = (0,30))
]