from pathlib import Path
import sys
sys.path.append(str(Path('.').absolute()))

from itertools import product
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from Utils import *
from tqdm import tqdm
from record_processors import *
from ddp_Utils import *

class ModelSaverOneEpoch:
    def __init__(self, model: nn.Module, model_folder, records, epoch=9, map_location=torch.device('cpu')):
        self.model = model
        self.model_folder = model_folder
        self.records = records
        self.epoch = epoch
        self.key = f'epoch_{epoch}'
        self.map_location = map_location
    
    def step(self):
        epoch = self.records['epoch']
        if epoch == self.epoch:
            print(f"Save model in epoch {epoch}.")
            ensure_path(self.model_folder)
            torch.save(self.model.state_dict(), f"{self.model_folder}/{self.key}.pth")
    
    def load(self):
        self.model.load_state_dict(torch.load(f'{self.model_folder}/{self.key}.pth', self.map_location))


def handle_na(y, label, n_classes):
    for i in range(len(y)):
        mask = ~torch.isnan(label[i])
        if n_classes[i] == 1:
            y[i] = y[i][mask]
        else:
            y[i] = y[i][mask, :]
        label[i] = label[i][mask]
    return y, label


def run_ddp(rank, world_size, config):
    nccl_port = config['nccl_port'] if 'nccl_port' in config else '12345'
    setup(rank, world_size, nccl_port)

    cpu = torch.device('cpu')
    task = config['task']

    id_base = config['id_base']
    processors = config['processors']
    savers_init = config['savers_init']
    optimizer_init = config['optimizer_init']
    scheduler_init = config['scheduler_init']
    loss_fn_init = config['loss_fn_init']

    train = config['train']
    batch_size = config['batch_size']
    num_train_epochs = config['num_train_epochs']
    dataset_class = config['dataset_class']
    model_class = config['model_class']

    folds = config['folds'] if 'folds' in config else range(5)
    
    task_path = f'{task}/exp-{id_base}'
    task_file_path = f'output/{task_path}/tasks/data.csv'
    pkl_dump(config, f'output/{task_path}/tasks/config.pkl')
    
    label_info = json_load(f'output/{task_path}/tasks/label_info.json')
    config = {**config, **label_info}

    n_classes = config['n_classes']
    label_cols = config['label_cols']


    df = pd.read_csv(task_file_path, low_memory=False)
    results = defaultdict(list)
    loss_fn = loss_fn_init(n_classes)
    for fold in folds:
        if rank == 0:
            print(f'task={task}, id_base={id_base}, fold={fold}')
        name = f"{task}-exp-{id_base}-fold{fold}"

        train_df = df[df['dataset'].isin([(fold + 1) % 5,  (fold + 2) % 5, (fold + 3) % 5])].copy()
        valid_df = df[df['dataset'] == (fold + 4) % 5].copy()
        if fold == 0:
            test_df = df[~df['dataset'].isin([(fold + 1) % 5,  (fold + 2) % 5, (fold + 3) % 5, (fold + 4) % 5])].copy()
        else:
            test_df = df[df['dataset'] == fold].copy()

        train_ds = dataset_class(train_df, 'train', config)
        valid_ds = dataset_class(valid_df, 'valid', config)
        test_ds = dataset_class(test_df, 'test', config)

        train_dl = DataLoader(train_ds, sampler=DS(train_ds, rank=rank), batch_size=batch_size, num_workers=world_size * 2, pin_memory=True)
        valid_dl = DataLoader(valid_ds, sampler=DS(valid_ds, rank=rank), batch_size=batch_size, num_workers=world_size * 2, pin_memory=True)
        test_dl = DataLoader(test_ds, sampler=DS(test_ds, rank=rank), batch_size=batch_size, num_workers=world_size * 2, pin_memory=True)

        model = model_class(config)
        model = model.to(rank)

        model_for_train = DDP(model, device_ids=[rank])

        records = defaultdict(list)
        records['fold'] = fold

        savers = []
        for saver_init in savers_init:
            savers.append(ModelSaver(model, f'models/{name}', records, saver_init[0], saver_init[1]))
        savers.append(ModelSaverOneEpoch(model, f'models/{name}', records, epoch=num_train_epochs - 1))

        if train:
            optimizer = optimizer_init(model)
            scheduler = scheduler_init(optimizer, num_train_epochs)
            for epoch in range(num_train_epochs):
                with Benchmark(f'Epoch {epoch}', print=rank == 0):
                    records['epoch'] = epoch
                    clear_records_epoch(records)

                    model_for_train.train()
                    t = tqdm(train_dl, position=0, leave=False) if rank == 0 else train_dl
                    for batch in t:
                        img = batch['img'].to(rank)
                        label = [l.to(rank) for l in batch['label']]
                        optimizer.zero_grad()
                        y, _ = model_for_train(img)
                        y, label = handle_na(y, label, n_classes)
                        loss = loss_fn(y, label)
                        # loss_bp = loss.mean()
                        loss.backward()
                        optimizer.step()

                        # loss = all_gather([loss])[0].detach().cpu()
                        if rank == 0:
                            t.set_postfix(loss=float(loss))
                            label = [l[~torch.isnan(l)] for l in label]
                            y = [yy[~torch.isnan(l)] for yy, l in zip(y, label)]

                            records['train-loss-list'].append([float(loss)])
                            records['train-y_true-list'].append([x.detach().cpu() for x in label])
                            records['train-y_pred-list'].append([x.detach().cpu() for x in y])
                            
                    model_for_train.eval()
                    with torch.no_grad():
                        t = tqdm(valid_dl, position=0, leave=False) if rank == 0 else valid_dl
                        for batch in t:
                            img = batch['img'].to(rank)
                            label = [l.to(rank) for l in batch['label']]
                            y, _ = model_for_train(img)
                            y, label = handle_na(y, label, n_classes)
                            loss = loss_fn(y, label)
                            # loss = all_gather([loss])[0].detach().cpu()

                            records['valid-loss-list'].append([float(loss)])
                            records['valid-y_true-list'].append([x.cpu() for x in label])
                            records['valid-y_pred-list'].append([x.cpu() for x in y])
                    to_print = []
                    scheduler.step(epoch)
                    if rank == 0:
                        for processor in processors:
                            key, value = processor(records)
                            records[key] = value
                            to_print.append(f'{key}={value:.4f}')
                        print(f'Epoch {epoch}: ' + ', '.join(to_print))

                        for saver in savers:
                            saver.step()

        model.predict = True
        for saver in savers:
            dist.barrier()
            saver.load()
            test_df_tmp = test_df.copy()

            preds = [[] for _ in range(len(n_classes))]
            image_paths = []
            embs = []
            model_for_train.eval()

            with torch.no_grad():
                t = tqdm(test_dl, position=0, leave=False) if rank == 0 else test_dl
                for batch in t:
                    img = batch['img'].to(rank)
                    image_path = batch['image_path']
                    y, emb = model_for_train(img)
                    for i in range(len(preds)):
                        if n_classes[i] != 1:
                            y[i] = torch.softmax(y[i], dim=-1)
                        preds[i].append(y[i].cpu())
                    image_paths.append(image_path)
                    embs.append(emb.cpu())
            
            image_paths = reduce(lambda x, y: x + y, image_paths)
            preds_df = pd.DataFrame(image_paths, columns=['image_path'])
            for i in range(len(preds)):
                preds[i] = torch.cat(preds[i], dim=0).numpy()
                if n_classes[i] == 1:
                    preds_df[f'{label_cols[i]}_prob_0'] = preds[i]
                else:
                    for j in range(1, n_classes[i]):
                        preds_df[f'{label_cols[i]}_prob_{j}'] = preds[i][:, j]
            preds_df = preds_df.merge(test_df_tmp, on='image_path', how='left')
            ensure_path(f'output/{task_path}/results/{saver.key}')
            preds_df.to_csv(f'output/{task_path}/results/{saver.key}/preds_fold_{fold}_rank_{rank}.csv', index=False)
            
            embs = torch.cat(embs, dim=0).numpy()
            np.savez_compressed(f'output/{task_path}/results/{saver.key}/embs_fold_{fold}_rank_{rank}.npz', emb=embs)
        model.predict = False
        model_for_train.to(cpu)
        model.to(cpu)
    print("All folds have been excuted")
    cleanup()