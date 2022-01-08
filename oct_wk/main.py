import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"


from pathlib import Path
import sys
sys.path.append(str(Path('.').absolute()))

from mains.exp_1114.oct_vf_model import OCTVFModel, MultiTaskLoss
from mains.exp_1114.oct_vf_dataset import OCTVFDataset
from mains.exp_1114.runner import *
from scheduler.cosine_lr import CosineLRScheduler
import torch.multiprocessing as mp


def optimizer_init(model):
    return Adam([
        {'params': model.backbone_parameters(), 'lr': 1e-4},
        {'params': model.head_parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)


def scheduler_init(optimizer, n_epoch):
    return CosineLRScheduler(optimizer, n_epoch)


def loss_fn_init(n_classes):
    return MultiTaskLoss(n_classes)


if __name__ == '__main__':
    print(torch.cuda.device_count())
    #label_info = json_load('output/1114/macula_num_r50_512/exp-pid/tasks/label_info.json')
    label_info = json_load('output/1230/disc_num_r50_512/exp-pid/tasks/label_info.json')
    config = {
        'task': '1230/disc_num_r50_512',
        'id_base': 'pid',
        'processors': [train_loss, valid_loss] + [partial(valid_auc, n_class=n_class, record_index=i) for i, n_class in enumerate(label_info['n_classes'])],
        'savers_init': [('valid-loss', min)],

        'train': True,
        'batch_size': 48,
        'num_train_epochs': 48,
        'dataset_class': OCTVFDataset,
        'model_class': OCTVFModel,
        'optimizer_init': optimizer_init,
        'scheduler_init': scheduler_init,
        'loss_fn_init': loss_fn_init,

        'image_root': Path('/home/octusr3/project/data_fast/new_slice'),
        'img_size': 512,
        'arch': 'resnet50',
        'folds': [0,1,2,3,4],
        #'folds': [0],
        'nccl_port': '10042'
    }
    mp.spawn(run_ddp, args=(4,config), nprocs=4, join=True)

