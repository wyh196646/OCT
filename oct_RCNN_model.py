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

    def __init__(self, classifier, roi_size=7, spatial_scale=16):
        super(OCT_ROI_head, self).__init__()
        self.feature,self.reg_layer=decom_ResNet50()
        self.vf_pred = nn.Linear(4096, 1)
        normal_init(self.vf_pred, 0, 0.01)
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)
 
    def forward(self, x, rois, roi_indices):
        roi_indices =torch.tensor(roi_indices).float()
        rois = torch.tensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        pool = self.roi(x, indices_and_rois)
        fc7 = self.classifier(pool)
        vf_predicted_value = self.vf_pred(fc7)
        
        return vf_predicted_value


def decom_ResNet50():
    model=models.resnet50(pretrained=True)
    feature=nn.Sequential(**list(model.children()[:-3]))
    reg_layer=nn.Sequential(
        model.avgpool,
        nn.Linear(512,1000)#输出的feature是1000
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
    