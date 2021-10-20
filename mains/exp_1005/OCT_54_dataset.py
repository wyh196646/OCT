from dataclasses import dataclass
from pandas.core.frame import DataFrame
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from pathlib import Path
import torch
from PIL import Image
import pandas as pd
from oct_Utils import *


@dataclass
class OCT_54_Dataset(Dataset):

    data:pd.DataFrame
    mode:str
    config:dict
    #total_slice:int

    def __post_init__(self):
        #self.countNums,_=calculate_position(str_to_np_mat(self.config['map_matrix']))
        #self.point_to_image_slice=reverse_dict(self.countNums)
        self.crop_size=self.config['crop_size']
        self.image_root=self.config['image_root']
        self.label_col=self.config['label_col']
        self.mode=self.mode
        self.trans = {
            'train': transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ColorJitter(brightness=0.3),
                transforms.Grayscale(num_output_channels=3),
                #transforms.RandomCrop((self.crop_size, self.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize([224,224]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([224,224]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }
       
    def read_image(self,path):
        img = Image.open(path).convert('RGB')
        img = self.trans[self.mode](img)
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        all_columns_name=self.data.columns.values
        img = self.read_image(str(self.image_root / self.data.loc[self.data.index[idx], 'image_path']))
        label = self.data.loc[self.data.index[idx],self.label_col]
        result = {
            'img': img,
            'label': torch.tensor(label, dtype=torch.float),
            #'label_position':str(all_columns_name[column+11])
        }
        return result
    











