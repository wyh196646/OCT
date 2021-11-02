from operator import imod
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"


from pathlib import Path
import sys
sys.path.append(str(Path('.').absolute()))
#sys.path.append("..")



from oct_RCNN_runner import *
from record_processors import *

from oct_RCNN_model import OCT_ROI_head
from oct_RCNN_dataset import OCT_RCNN_Dataset

if __name__ == '__main__':
    config = {
        'task': '1102/disc_num_r50_512',
        'id_base': 'pid',
        'processors': [train_loss, valid_loss], 
        'savers_init': [('valid-loss', min)],
        'batch_size': 64,#batch_size勉强调到64才可以用，不会爆内存了
        'num_train_epochs': 50,
        'parallel': True,
        'mode':'train',
        'label_col':'num',
        'train':True,
        'image_size':512,#配置用来做数据集输入的图片大小,一般裁剪出来的图片都是正方形的，所以传入一个参数即可
        'dataset_class': OCT_RCNN_Dataset,
        'model_class': OCT_ROI_head,
        'crop_size': 320,
        'image_root': Path('/home/octusr3/project/data_fast/512'),
        'map_matrix':'''[ [0  0  0  0  0  0  0  0  0  0]
                        [0  0  0  46 44 43 42 0  0  0]
                        [0  0  46 45 44 43 41 43 0  0]
                        [0  45 46 45 45 45 42 42 41 0]
                        [46 46 47 46 50 50 50 3  41 0]
                        [9  9  9  6  8  6  7  3  16 0]
                        [0  10 8  7  6  8  9  13 17 0]
                        [0  0  8  8  10 10 12 15 0  0]
                        [0  0  0  9  10 11 14 0  0  0]
                        [0  0  0  0  0  0  0  0  0  0]]''',

        'arch': 'resnet50'  #中间两个tag为3的点是视野盲区，无法推断
    }
    run(config)

