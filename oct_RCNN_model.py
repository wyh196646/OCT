import numpy as np
import pandas as pd
from torch._C import device
from torch.types import Device
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

    def __init__(self, spatial_scale=16):#这里的spatial_scale直接弄错了，导致代码逻辑出现了巨大的问题，本来应该是十六分之一比较合理，这样才有向下缩小的倍数
        #弄成16，一下把所有的特征点都囊括进去了，所以模型就不收敛。
        super(OCT_ROI_head, self).__init__()#roi_size作为超参数进行调节
        self.loss_fn=nn.MSELoss(reduction='none')
        self.feature,self.behind_layer,self.classfier=self.decom_ResNet50()
        #self.vf_pred = nn.Linear(4096, 54)
        #self.normal_init(self.vf_pred, 0, 0.01)
        self.spatial_scale = spatial_scale
        
        #一次取batch_size大小的数据方式有待进一步学习Pytorch的代码，理清结构，才能平衡batch的关系
        #仍然要读旧的代码
    def forward(self,img, rois,roi_size_h_ratio=0.25,roi_size_w_ratio=0.25,label=None):#先按照输出维度roi_size=16来进行调整
        x=self.feature(img)
        n,_,hh,ww=x.shape#n就是batch_size，所以在写模型的的时候都要考虑到batchsize的维度
        self.roi_h_size=hh*roi_size_h_ratio
        self.roi_w_size=ww*roi_size_w_ratio
        self.roi = RoIPool((self.roi_h_size, self.roi_w_size),1./self.spatial_scale)
    
        roi_indices=list()
        for i in range(n):#整合roi_indices是一个需要理解的过程，框的维度 batch*54*4，roi_indices就应该是batch*54*1，所以比如第一个batch数据，就要有一个54*1大小的全1矩阵来拼接才可以
            batch_index = i * np.ones((54,), dtype=np.int32)#所以下面的代码可以work
            roi_indices.append(batch_index)
        roi_indices = torch.tensor(np.concatenate(roi_indices, axis=0)).to(rois.device)

        roi_indices=roi_indices.view(n,54,1)#128*54
        #print(roi_indices.shape)
        #print(rois.shape)#重新调配维度
        indices_and_rois = torch.cat([roi_indices, rois], dim=2).to(rois.device)#注意这个rois_indice[: ,None]会升维
        #indices_and_rois=xy_indices_and_rois.to(rois.device,dtype=torch.float) #batch*54*4和batch*54*1进行合并
        pool = self.roi(x, indices_and_rois)#x是原始特征图，indices是组合以后的anchor，输出的维度应该是batchsize*54*channel*roi_size*roi_size,所以要适配后面的问题
        pool=pool.view(pool.shape[0]*54,512,self.roi_h_size,roi_size_w_ratio)# batchsize*54*512*7*7 ，这里进行维度修正
        pool=self.behind_layer(pool).squeeze()
        
        fc7 = self.classfier(pool)
        #vf_predicted_value = self.vf_pred(fc7)#vf_predicted_value的维度：54*1 proposal数量决定的第一维度
        #所以尽量让这里的label进行维度匹配
        #vf_predicted_value=vf_predicted_value.squeeze(1)
        
        loss=None
        if label is not None:
            label=label.squeeze(1)
            loss=self.loss_fn(vf_predicted_value,label)#仍旧相当于一次性预测了原有的54个点
        #print(loss.shape)
        return vf_predicted_value,loss#标记输出维度是一个非常好的方法，有助于理清思路，快速带入代码


    def backbone_parameters(self):
        return self.feature.parameters()
    
    def head_parameters(self):
        return self.classfier.parameters()

    
    def decom_ResNet50(self):
        model=models.resnet50(pretrained=True)
        feature=nn.Sequential(*list(model.children())[:-4])#适用childen方法，可以将预训练的参数带过去，
        behind_layer=nn.Sequential(#roi输出的维度是 1024*7*7
            #输出的feature是1000,根据Resnet网络结构分析出来的维度
            *list(model.children())[-4:-1]
        )
        reg_layer=nn.Linear(2048,54)
        return feature,behind_layer,reg_layer
    #a,b,c=decom_ResNet50()


    def normal_init(self,m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()
    