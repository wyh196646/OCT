import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import cv2
import os
import re
import pickle
import math
from pathlib import *
import multiprocessing
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from collections import *
from itertools import *
from functools import *
from sklearn.metrics import *
from scipy.stats import *
import pandas as pd
import seaborn as sns
import hashlib
from PIL import Image
from datastatics import *
from oct_Utils import *
import itertools
from torchvision.ops import RoIPool
import torchvision.models as models
from torchsummary import summary

class OCT_ROI_head(nn.Module):

    def __init__(self,config,classifier,roi_size=7, spatial_scale=16):
        super(OCT_ROI_head, self).__init__()
        self.loss_fn=nn.MSELoss(reduction='none')
        self.config = config
        self.feature,self.reg_layer=self.decom_ResNet50()
        self.classifier=classifier
        self.vf_pred = nn.Linear(4096, 1)
        self.normal_init(self.vf_pred, 0, 0.01)
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)
    

    def forward(self, x, rois, roi_indices,label=None):
        roi_indices =torch.tensor(roi_indices).float()
        rois = torch.tensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois=xy_indices_and_rois
        pool = self.roi(x, indices_and_rois)#x是原始特征图，indices是组合以后的anchor
        pool=pool.view(pool.size(0),-1)
        fc7 = self.classifier(pool)
        vf_predicted_value = self.vf_pred(fc7)
        loss=None
        if label is not None:
            loss=self.loss_fn(vf_predicted_value,label)
        return vf_predicted_value,loss#标记输出维度是一个非常好的方法，有助于理清思路，快速带入代码



    def decom_ResNet50():
        model=models.resnet50(pretrained=True)
        feature=nn.Sequential(**list(model.children()[:-3]))
        reg_layer=nn.Sequential(
            nn.Linear(512*7*7,4096)#输出的feature是1000
        )
        return feature,reg_layer


    def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()
    