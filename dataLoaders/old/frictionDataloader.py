from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, os.path
import torch
from PIL import *
import glob
from torchvision import transforms
import pandas as pd
from scipy import signal
import cv2





class FrictionLoader(Dataset):


    def __init__(self, datasetConfig, transform_in = None, transform_ou = None):

        super().__init__()
        self.datasetConfig = datasetConfig

        self.sig_add = "../onedrive/General/{}Test{}/sorted/{}/{}.csv".format(self.datasetConfig["season"], self.datasetConfig["year"], self.datasetConfig["date"] + "_" + self.datasetConfig["hour"], self.datasetConfig["date"])
        self.vid_add = "../onedrive/General/{}Test{}/sorted/{}".format(self.datasetConfig["season"], self.datasetConfig["year"], self.datasetConfig["date"] + "_" + self.datasetConfig["hour"])

        
        self.signals = pd.read_csv(self.sig_add)
        self.signals["date_time"] = pd.to_datetime(self.signals["date_time"])
        self.signals = self.signals.set_index('date_time')
        self.signals = self.signals.loc[~self.signals.index.duplicated(keep='first')]
        self.signals.sort_values(by='date_time', inplace=True)
        self.signals.reset_index(inplace=True)
        
        [a,b] = signal.butter(10, 0.02)
        self.signals['IsoVsLongitudinalAcceleration'] = signal.filtfilt(a, b, self.signals['IsoVsLongitudinalAcceleration'])
        self.signals["avg_mu"] = self.signals['IsoVsLongitudinalAcceleration']/9.82
        
        
        filename = glob.glob(os.path.join(self.vid_add,"*.mp4"))[0]
        self.cap = cv2.VideoCapture(filename)
        
        self.transform_in = transform_in
        self.transform_ou = transform_ou

    def __len__(self):
        return len(self.signals[self.signals["FrameNum"]>0])

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame = self.retFrame(self.signals[self.signals["FrameNum"]>0][["FrameNum"]].iloc[idx].item())
        mu = self.signals[self.signals["FrameNum"]>0][["avg_mu"]].iloc[idx].item()

        if self.transform_in:
            img = self.transform_in(frame)
        if self.transform_ou:
            pass

        return {'image': img.float(), 'label': mu}


    def retFrame(self, indexFrame):

        self.cap.set(1, indexFrame)
        ret, frame = self.cap.read()

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
