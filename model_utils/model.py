import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch_optimizer as optim
from torch.nn.modules.loss import _Loss

import cv2
import time
import os, sys
import pandas as pd
import albumentations as A
import numpy as np
from glob import glob
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)
import time
from sklearn import metrics

from torch.utils.tensorboard import SummaryWriter
from scipy.special import softmax

import timm

from segmentation_models_pytorch.losses import DiceLoss



class GHDModel(nn.Module):   
    def __init__(self, C):
        super(GHDModel, self).__init__()

        self.L = 2048 #2048 # 256  # num_output_feature of the encoder_2
        self.D = 224
        self.K = 1

        self.C = C


        self.encoder = timm.models.hrnet_w18(pretrained=False,num_classes=0, global_pool='')
        feature_dim = self.L # num_output_feature of the encoder

        self.seg = nn.Sequential(
            # nn.Conv2d(feature_dim,4,kernel_size=1,stride=1),
            nn.ConvTranspose2d(in_channels=feature_dim, out_channels=4, kernel_size=2, stride=2),
            nn.Softmax2d(),

        )

        self.encoder_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.C)

        self.classifier_linears = nn.ModuleList([nn.Linear(self.L*self.K, 1) for _ in range(self.C)])

       
    def weight_average(self, x, w,num_instances):
        w = w.view(-1,1)
        M = torch.multiply(x,w)
        M = M.view(-1, num_instances, self.L)  # bs x N x L
        M = torch.sum(M,dim=1)  # bs x L

        return M

    def softmax_one(self, x, dim=None):
        #subtract the max for stability
        x = x - x.max(dim=dim, keepdim=True).values
        #compute exponentials
        exp_x = torch.exp(x)
        #compute softmax values and add on in the denominator
        return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))

    def forward(self, x):
        # x: bs x N x C x W x W
        bs, num_instances, ch, w, h = x.shape

        x = x.view(bs*num_instances, ch, w, h) # x: N bs x C x W x W

        x = self.encoder.forward_features(x) # x: N bs x C' x W' x W'

        y = self.seg(x)  # 32 --> 128 

        x = self.encoder_2(x) # x: N bs x self.L  

        A_V = self.attention_V(x)  # N bs x self.D
        A_U = self.attention_U(x)  # N bs x self.D

        A = self.attention_weights(A_V * A_U) # element wise multiplication # N bs x self.C

        A = A.view(bs, num_instances, self.C)  # bs x N x  C
        # A = F.softmax(A, dim=1).view(-1,self.C) 
        A = self.softmax_one(A, dim=1).view(-1,self.C) 


        x = [self.weight_average(x,i,num_instances) for i in torch.split(A, split_size_or_sections=1, dim=1)]
        x = [self.classifier_linears[i](x[i]) for i in range(self.C)]  # [] bs x 1
        Y_prob = torch.cat(x,1)

        return Y_prob, y

def load_model(C, weight_path=None,strict=False, device=torch.device("cuda")):
    print('loading model')
    model = GHDModel(C)
    # load pretrain
    if weight_path is not None:
        print ('loading pretrained model {}'.format(weight_path))
        stdict = torch.load(weight_path,map_location=device)#['state_dict']
        new_stdict = {}
        if strict:
            prefix = ''
        else:
            prefix = 'encoder.'
        for i,(k,v) in enumerate(stdict.items()):
            if 'module.' in k:
                new_stdict[prefix + k[7:]] = stdict[k]
            else:
                new_stdict[prefix + k] = stdict[k]
        missing_keys = [i for i in model.state_dict().keys() if i not in new_stdict.keys()]
        print('missing_keys',missing_keys)
        model.load_state_dict(
            new_stdict, strict=strict
        )
    
    model = model.to(device)
    model = nn.DataParallel(model)
    # print(model)
    return model
