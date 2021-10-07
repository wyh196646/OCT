from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from pathlib import Path
import torch
from PIL import Image
import pandas as pd
from oct_Utils import *


@dataclass
class OCTVFDataset(Dataset):
    data: pd.DataFrame
    mode: str
    config: dict  # crop_size

    def __post_init__(self):
        self.image_root = self.config['image_root']
        self.crop_size = self.config['crop_size']
        self.label_col = self.config['label_col']
        self.pdp_col = self.config['pdp_col']
        self.loss_weights_mapping = self.config['loss_weights_mapping']
        self.valid_mask = str_to_np_mat(self.config['valid_mask'], dtype=int) == 1

        self.trans = {
            'train': transforms.Compose([
                transforms.ColorJitter(brightness=0.3),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop((self.crop_size, self.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }

    def read_image(self, path):
        img = Image.open(path).convert('RGB')
        img = self.trans[self.mode](img)
        return img

    def read_np_mat(self, idx, col):
        mat = str_to_np_mat(self.data.loc[self.data.index[idx], col], dtype=float)
        eye = self.data.loc[self.data.index[idx], 'eye']
        if eye == 'OS':
            mat = np.flip(mat, axis=1)
        result = mat[self.valid_mask]
        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.read_image(str(self.image_root / self.data.loc[self.data.index[idx], 'image_path']))
        label = self.read_np_mat(idx, self.label_col)
        pdp = self.read_np_mat(idx, self.pdp_col)
        loss_weights = pdp.copy()
        for i, v in enumerate(self.loss_weights_mapping):
            loss_weights[pdp == i] = v
        result = {
            'img': img,
            'img_path': str(self.image_root / self.data.loc[self.data.index[idx], 'image_path']),
            'label': torch.tensor(label, dtype=torch.float),
            'pdp': torch.tensor(pdp, dtype=torch.long),
            'loss_weights': torch.tensor(loss_weights, dtype=torch.float),
        }
        return result



if __name__ == '__main__':
    df = pd.read_csv('/home/octusr2/projects/data_fast/csv/0820.csv')

    config = {
        'crop_size': 320,
        'image_root': Path('/home/octusr2/projects/data_fast/proceeded/cp_projection/380'),
        'label_col': 'num',
        'valid_mask': '''[[0 0 0 0 0 0 0 0 0 0]
                        [0 0 0 1 1 1 1 0 0 0]
                        [0 0 1 1 1 1 1 1 0 0]
                        [0 1 1 1 1 1 1 1 1 0]
                        [1 1 1 1 1 1 1 1 1 0]
                        [1 1 1 1 1 1 1 1 1 0]
                        [0 1 1 1 1 1 1 1 1 0]
                        [0 0 1 1 1 1 1 1 0 0]
                        [0 0 0 1 1 1 1 0 0 0]
                        [0 0 0 0 0 0 0 0 0 0]]'''
    }
    ds = OCTVFDataset(df, 'train', config)
    for batch in ds:
        img = batch['img']
        img_path = batch['img_path']
        label = batch['label']
        pass
