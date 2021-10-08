from operator import imod
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


from pathlib import Path
import sys
sys.path.append(str(Path('.').absolute()))
#sys.path.append("..")

from oct_vf_dataset import OCTVFDataset
#from oct_vf_model import OCTVFModel
from runner54 import *
from record_processors import *
from OCT_54_dataset import OCT_54_Dataset
from OCT_54_vf_model import OCTVF54Model

if __name__ == '__main__':
    config = {
        'task': '1005/disc_num_r50_380',
        'id_base': 'pid',
        'processors': [train_loss, valid_loss], 
        'savers_init': [('valid-loss', min)],
        'batch_size': 512,
        'num_train_epochs': 30,
        'parallel': True,
        'mode':'train',
        'label_col':'value',
        'train':True,
        'dataset_class': OCT_54_Dataset,
        'model_class': OCTVF54Model,
        'crop_size': 320,
        'self.col_name':'value',
        'image_root': Path('/home/octusr3/project/data_fast/54'),


        'map_matrix':'''[ [0  0  0  0  0  0  0  0  0  0]
                        [0  0  0  46 44 43 42 0  0  0]s
                        [0  0  46 45 44 43 41 43 0  0]
                        [0  45 46 45 45 45 42 42 41 0]
                        [46 46 47 46 50 50 50 3  41 0]
                        [9  9  9  6  8  6  7  3  16 0]
                        [0  10 8  7  6  8  9  13 17 0]
                        [0  0  8  8  10 10 12 15 0  0]
                        [0  0  0  9  10 11 14 0  0  0]
                        [0  0  0  0  0  0  0  0  0  0]]''',

        'arch': 'resnet50'#中间两个tag为3的点是视野盲区，无法推断
    }
    run(config)
