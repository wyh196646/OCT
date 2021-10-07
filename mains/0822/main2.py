import os
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"


from pathlib import Path
import sys
sys.path.append(str(Path('.').absolute()))

from oct_vf_dataset import OCTVFDataset
from oct_vf_model import OCTVFModel
from runner import *


if __name__ == '__main__':
    config = {
        'task': '0822/macula_num_r50_380',
        'id_base': 'pid',
        'processors': [train_loss, valid_loss] + [partial(valid_auc, n_class=1, record_index=i) for i in range(54)],
        'savers_init': [('valid-loss', min)],

        'train': True,
        'batch_size': 64,
        'num_train_epochs': 30,
        'parallel': True,
        'dataset_class': OCTVFDataset,
        'model_class': OCTVFModel,

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
                        [0 0 0 0 0 0 0 0 0 0]]''',

        'arch': 'resnet50'
    }
    run(config)
