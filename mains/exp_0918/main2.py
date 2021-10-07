import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"


from pathlib import Path
import sys
sys.path.append(str(Path('.').absolute()))

from mains.exp_0831.oct_vf_dataset import OCTVFDataset
from mains.exp_0831.oct_vf_model import OCTVFModel
from mains.exp_0831.runner import *


if __name__ == '__main__':
    config = {
        'task': '0918/macula_num_r50_380',
        'id_base': 'pid',
        'processors': [train_loss, valid_loss, normal_valid_loss, abnormal_valid_loss] + [partial(valid_auc, n_class=1, record_index=i) for i in range(54)],
        'savers_init': [('valid-loss', min),('n-valid-loss', min),('abn-valid-loss', min)],

        'train': True,
        'batch_size': 256,
        'num_train_epochs': 50,
        'parallel': True,
        'dataset_class': OCTVFDataset,
        'model_class': OCTVFModel,

        'crop_size': 320,
        'image_root': Path('/home/octusr2/projects/data_fast/proceeded/cp_projection_backup/380'),
        'label_col': 'num',
        'pdp_col': 'pd_prob',
        'loss_weights_mapping': [1, 1, 2, 3, 4, 5],
        # 'loss_weights_mapping': [1, 1, 1, 1, 1, 1,],

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
