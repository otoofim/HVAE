import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.distributions import Normal, Independent, kl
import torchvision
import torchvision.transforms as T
from torch.distributions import Normal, Independent, kl, MultivariateNormal
from UNetBlocks import *



class UNet(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, num_samples, num_classes, LatentVarSize = 6, posterior = False):
        super(UNet, self).__init__()
        #Vars init
        self.LatentVarSize = LatentVarSize
        self.posterior = posterior
        self.num_samples = num_samples
        self.num_classes = num_classes

        
        #architecture
        self.DownConvBlock1 = DownConvBlock(input_dim = 6 if posterior else 3, output_dim = 128, ResLayers = 2, initializers = None, padding = 1)
        self.DownConvBlock2 = DownConvBlock(input_dim = 128, output_dim = 256, ResLayers = 2, initializers = None, padding = 1)
        self.DownConvBlock3 = DownConvBlock(input_dim = 256, output_dim = 512, ResLayers = 2, initializers = None, padding = 1)
        self.DownConvBlock4 = DownConvBlock(input_dim = 512, output_dim = 256, ResLayers = 2, latent_dim = LatentVarSize, initializers = None, padding = 1)

        self.UpConvBlock1 = UpConvBlock(input_dim = 256, output_dim = 512, ResLayers = 2, initializers = None, padding = 1)
        self.UpConvBlock2 = UpConvBlock(input_dim = (512 * 2) + LatentVarSize, output_dim = 256, ResLayers = 2, latent_dim = LatentVarSize, initializers = None, padding = 1)
        self.UpConvBlock3 = UpConvBlock(input_dim = (256 * 2) + LatentVarSize, output_dim = 128, ResLayers = 2, latent_dim = LatentVarSize, initializers = None, padding = 1)
        self.UpConvBlock4 = UpConvBlock(input_dim = (128 * 2) + LatentVarSize, output_dim = self.num_classes, ResLayers = 2, initializers = None, padding = 1)
        self.softmax = nn.Softmax(dim=1)

        #loss functions
        self.criterion = nn.BCEWithLogitsLoss(size_average = False, reduction = None, reduce = False)
        
    def forward(self, inputFeatures):
        
        dists = {}
        encoderOuts = {}
        
        encoderOuts["out1"] = self.DownConvBlock1(inputFeatures)
        encoderOuts["out2"] = self.DownConvBlock2(encoderOuts["out1"])
        encoderOuts["out3"] = self.DownConvBlock3(encoderOuts["out2"])
        encoderOuts["out4"], dists["dist1"] = self.DownConvBlock4(encoderOuts["out3"])

        out = self.UpConvBlock1(encoderOuts["out4"])
        latent1 = torch.nn.Upsample(size=encoderOuts["out3"].shape[2:], mode='nearest')(dists["dist1"].rsample())
        out = torch.cat((encoderOuts["out3"], out, latent1), 1)

        out, dists["dist2"] = self.UpConvBlock2(out)
        latent2 = torch.nn.Upsample(size=encoderOuts["out2"].shape[2:], mode='nearest')(dists["dist2"].rsample())
        out = torch.cat((encoderOuts["out2"], out, latent2), 1)

        out, dists["dist3"] = self.UpConvBlock3(out)
        latent3 =  torch.nn.Upsample(size=encoderOuts["out1"].shape[2:], mode='nearest')(dists["dist3"].rsample())
        out = torch.cat((encoderOuts["out1"], out, latent3), 1)
                
        if self.posterior:
            return dists
        
        out = self.UpConvBlock4(out)
        out = self.softmax(out)
        
        return out, dists
    
    def inference(self, inputFeatures):
        
        with torch.no_grad():
            dists = {}
            encoderOuts = {}
            samples = []

            encoderOuts["out1"] = self.DownConvBlock1(inputFeatures)
            encoderOuts["out2"] = self.DownConvBlock2(encoderOuts["out1"])
            encoderOuts["out3"] = self.DownConvBlock3(encoderOuts["out2"])
            encoderOuts["out4"], dists["dist1"] = self.DownConvBlock4(encoderOuts["out3"])

            for _ in range(self.num_samples):

                out = self.UpConvBlock1(encoderOuts["out4"])
                latent1 = torch.nn.Upsample(size=encoderOuts["out3"].shape[2:], mode='nearest')(dists["dist1"].rsample())
                out = torch.cat((encoderOuts["out3"], out, latent1), 1)
                
                if "dist2" not in dists.keys():
                    out, dists["dist2"] = self.UpConvBlock2(out)
                else:
                    out, _ = self.UpConvBlock2(out)
                latent2 = torch.nn.Upsample(size=encoderOuts["out2"].shape[2:], mode='nearest')(dists["dist2"].rsample())
                out = torch.cat((encoderOuts["out2"], out, latent2), 1)

                
                if "dist3" not in dists.keys():
                    out, dists["dist3"] = self.UpConvBlock3(out)
                else:
                    out, _ = self.UpConvBlock3(out)
                latent3 =  torch.nn.Upsample(size=encoderOuts["out1"].shape[2:], mode='nearest')(dists["dist3"].rsample())
                out = torch.cat((encoderOuts["out1"], out, latent3), 1)

                out = self.UpConvBlock4(out)
                out = self.softmax(out)

                samples.append(out)
        
        
        return torch.stack(samples), dists

    
