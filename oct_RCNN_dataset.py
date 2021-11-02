from dataclasses import dataclass
from pyexpat import model
from pandas.core.frame import DataFrame
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from pathlib import Path
import torch
from PIL import Image
import pandas as pd
from oct_Utils import *




@dataclass
class OCT_RCNN_Dataset(Dataset):

    data:pd.DataFrame
    mode:str
    config:dict


    def __post_init__(self):
        self.crop_size=self.config['crop_size']
        self.image_root=self.config['image_root']
        self.label_col=self.config['label_col']
        self.image_size=self.config['image_size']
        self.trans = {
            'train': transforms.Compose([
                #transforms.Resize([224,224]),
                transforms.ColorJitter(brightness=0.3),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                #transforms.Resize([224,224]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                #transforms.Resize([224,224]),
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

    def read_np_mat(self, idx, col):
        mat = str_to_np_mat(self.data.loc[self.data.index[idx], col], dtype=float)
        eye = self.data.loc[self.data.index[idx], 'eye']
        if eye == 'OS':
            mat = np.flip(mat, axis=1)
        return mat

    def generate_vf_proposal(self):
        '''
        这里需要对边界框作复映射，以此来保证最终proposal区域够54个点
        '''
        vf_dict,_=calculate_position(str_to_np_mat(self.config['map_matrix']))
        valid_slice=[]
        valid_position=[]
        for key,value in vf_dict.items():
            for tem in value:
                valid_slice.append(key)
                valid_position.append(tem)#valid_slice.append(len(value)*[key])已经扩充到了54点，中间包含重复值的slice
        centre_point_x=np.linspace((self.image_size/54)/2,self.image_size-(self.image_size/54)/2,54)
        centre_point_y=np.repeat(self.image_size/2,54)
        w,h=self.image_size/54,self.image_size
        temp=[(centre_point_x-w/2).reshape(54,-1),(centre_point_y-h/2).reshape(54,-1),(centre_point_x+w/2).reshape(54,-1),(centre_point_y+h/2).reshape(54,-1)]
        proposal=np.concatenate(temp,axis=1)
        #return valid_slice,这里-1 是因为slice标记是1-54,映射到anchor序列应该是0-53
        return proposal[np.array((valid_slice),dtype=int)-1],valid_slice,valid_position#提取出有效proposal区域

    def get_proposal_label(self,mat,valid_position):
        #根据序列标识进行测试，最终得到的结果
        #坐标的序列都是从0-9，所以 直接映射就可以
        res=[]
        for i in valid_position:
            res.append(mat[i[0]][i[1]])
        return np.array(res)

  
        
    def __getitem__(self, idx):
        all_columns_name=self.data.columns.values
        img = self.read_image(str(self.image_root / self.data.loc[self.data.index[idx], 'image_path']))
       
        label=self.read_np_mat(idx,self.label_col)
        proposal,valid_slice,valid_position=self.generate_vf_proposal()#直接通过类属性访问即可
        label=self.get_proposal_label(label,valid_position)
        result = {
            'img': img,
            'proposal':proposal,#生成的有效坐标框区域,label是坐标对应的框
            'label': torch.tensor(label, dtype=torch.float),
        }
        return result
    
# if __name__ == '__main__':
#     df = pd.read_csv('/home/octusr3/project/oct/data.csv')

#     config = {
#         'crop_size': 320,
#         'image_root': Path('/home/octusr2/projects/data_fast/proceeded/cp_projection/380'),
#         'label_col': 'num',
#         'valid_mask': '''[[0 0 0 0 0 0 0 0 0 0]
#                         [0 0 0 1 1 1 1 0 0 0]
#                         [0 0 1 1 1 1 1 1 0 0]
#                         [0 1 1 1 1 1 1 1 1 0]
#                         [1 1 1 1 1 1 1 1 1 0]
#                         [1 1 1 1 1 1 1 1 1 0]
#                         [0 1 1 1 1 1 1 1 1 0]
#                         [0 0 1 1 1 1 1 1 0 0]
#                         [0 0 0 1 1 1 1 0 0 0]
#                         [0 0 0 0 0 0 0 0 0 0]]'''
#     }
#     ds = OCT_RCNN_Dataset(df, 'train', config)
#     for batch in ds:
#         img = batch['img']
#         img_path = batch['img_path']
#         label = batch['label']
#         print(label.shape)
        
#         pass
    
#     print(ds)
#     print('===========')



















