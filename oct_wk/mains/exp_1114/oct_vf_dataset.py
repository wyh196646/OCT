from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from pathlib import Path
import torch
from PIL import Image
import pandas as pd
from oct_Utils import *
from augment.auto_augment import transforms_imagenet_train, transforms_imagenet_eval


@dataclass
class OCTVFDataset(Dataset):
    data: pd.DataFrame
    mode: str
    config: dict

    def __post_init__(self):
        self.image_root = self.config['image_root']
        self.img_size = self.config['img_size']
        self.label_cols = self.config['label_cols']
        self.trans = {
            'train': self.augmentation(self.img_size, train=True),
            'valid': self.augmentation(self.img_size, train=False),
            'test': self.augmentation(self.img_size, train=False),
        }

    def augmentation(self, image_size, train=True):
        if train:
            return transforms_imagenet_train(image_size, auto_augment='rand-m9-mstd0.5-inc1')
        return transforms_imagenet_eval(image_size)

    def read_image(self, path):
        img = self.trans[self.mode](Image.open(path).convert('RGB'))
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.read_image(str(self.image_root / self.data.loc[self.data.index[idx], 'image_path']))
        result = {
            'img': img,
            'image_path': self.data.loc[self.data.index[idx], 'image_path'],
            'label': [torch.tensor(self.data.loc[self.data.index[idx], col], dtype=torch.float) for col in self.label_cols]
        }
        return result