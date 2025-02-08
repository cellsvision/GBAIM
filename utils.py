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

# import timm

from segmentation_models_pytorch.losses import DiceLoss

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def get_h_label(x):
    x1 = np.round(x)
    if max(x1)==0:
        return np.argmax(x)
    else:
        for i in list(range(len(x1)))[::-1]:
            if x1[i]==1:
                return i

class MULLOSS(_Loss):
    def __init__(self,lossbce,lossdice) -> None:
        super(MULLOSS, self).__init__()
        self.lossbce = lossbce
        self.lossdice = lossdice
        
    def forward(self, 
                y_pred: torch.Tensor, 
                y_true: torch.Tensor,
                mask_pred: torch.Tensor,
                mask ) -> torch.Tensor:
        bce = self.lossbce(y_pred, y_true)
        dice = self.lossdice(mask_pred, mask)
        total_loss = bce*1.0 + dice*0.1
        return total_loss,bce,dice





def get_final_label(x,threshs=None):
    if threshs==None:
        x1 = np.round(x)
    elif not isinstance(threshs,list):
        x1 = (x>=threshs).astype(np.int)
    else:
        x1 = np.zeros_like(x)
        for i,th in enumerate(threshs):
            # x1[:,i] = [1 if p>th else 0 for p in x[:,i]]
            x1[r,0] = 1

    for r in range(len(x1)):
        if np.max(x1[r])==0:
            # x1[r][np.argmax(x[r])] = 1
            x1[r,0] = 1
    return x1 