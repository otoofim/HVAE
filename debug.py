from ProUNet import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.distributions import Normal, Independent, kl
import torchvision
import torchvision.transforms as T
from torch.distributions import Normal, Independent, kl, MultivariateNormal



input_test1 = torch.rand((2,3,512,512))
seg_test1 = torch.rand((2,3,512,512))
model = ProUNet()
optimizer = torch.optim.Adam(model.parameters(), lr =  0.00001, weight_decay = 0.00001)


seg, priorDists, posteriorDists = model(input_test1,seg_test1)



loss = model.elbo_loss(input_test1, seg_test1, priorDists, posteriorDists)
print(loss)

