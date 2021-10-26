import sys
from itertools import product
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from Utils import *
from tqdm import tqdm
from record_processors import *

sys.path.append("..")

def run(config, device=torch.device('cuda')):
    cpu = torch.device('cpu')

    task = config['task']
    id_base = config['id_base']
    processors = config['processors']
    savers_init = config['savers_init']
    
    train = config['train']
    batch_size = config['batch_size']
    num_train_epochs = config['num_train_epochs']
    print(num_train_epochs)
    parallel = config['parallel']
    dataset_class = config['dataset_class']
    model_class = config['model_class']
    folds = config['folds'] if 'folds' in config else range(5)
    task_path = f'{task}/exp-{id_base}'
    task_file_path = f'output/{task_path}/tasks/data.csv'
    df = pd.read_csv(task_file_path, low_memory=False)
    #print(df)
    results = defaultdict(list)
    for fold in folds:
        print(f'task={task}, id_base={id_base}, fold={fold}')
        name = f"{task}-exp-{id_base}-fold{fold}"

        train_df = df[(df['dataset'] != (fold - 1) % 5) & (df['dataset'] != fold)].copy()
        valid_df = df[(df['dataset'] == (fold - 1) % 5)].copy()
        test_df = df[(df['dataset'] == fold)].copy()

        train_ds = dataset_class(train_df, 'train', config)
        valid_ds = dataset_class(valid_df, 'valid', config)
        test_ds = dataset_class(test_df, 'test', config)

        train_dl = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=batch_size, num_workers=32)
        valid_dl = DataLoader(valid_ds, sampler=SequentialSampler(valid_ds), batch_size=batch_size, num_workers=32)
        test_dl = DataLoader(test_ds, sampler=SequentialSampler(test_ds), batch_size=batch_size, num_workers=32)

        model = model_class(config)
        model = model.to(device)

        if parallel:
            model_for_train = nn.DataParallel(model)
        else:
            model_for_train = model
        records = defaultdict(list)
        records['fold'] = fold  

        savers = []
        for saver_init in savers_init:
            savers.append(ModelSaver(model, f'models/{name}', records, saver_init[0], saver_init[1]))

        if train:


            optimizer = Adam([
                {'params': model.backbone_parameters(), 'lr':1e-5},
                {'params': model.head_parameters(), 'lr': 1e-4}
            ], weight_decay=0.01)

            scheduler = lr_scheduler.StepLR(optimizer, step_size=batch_size // 2, gamma=0.1)

            for epoch in range(num_train_epochs):
                with Benchmark(f'Epoch {epoch}'):
                    records['epoch'] = epoch
                    clear_records_epoch(records)

                    model_for_train.train()
                    with tqdm(train_dl, leave=False, file=sys.stdout) as t:
                        for batch in t:
                            img = batch['img'].to(device)
                            label = batch['label'].to(device).unsqueeze(1)
                            #label = label.unsqueeze(1)
                            optimizer.zero_grad()
                            y, loss = model_for_train(img, label)
                            loss_bp=torch.mean(loss)    
                            loss_bp.backward()
                            optimizer.step()
                            t.set_postfix(loss=float(loss_bp))
                            records['train-loss-list'].append([float(loss_bp)])
                    model_for_train.eval()
                    with torch.no_grad():
                        with tqdm(valid_dl, leave=False, file=sys.stdout) as t:
                            for batch in t:
                                img = batch['img'].to(device)
                                label = batch['label'].to(device).unsqueeze(1)
                                optimizer.zero_grad()#用验证集来评估模型的好坏，最后test做结果
                                loss_bp = torch.mean(loss)
                                y, loss= model_for_train(img, label)
                                records['valid-loss-list'].append([float(loss_bp)])
              
                    to_print = []
                    for processor in processors:
                        key, value = processor(records)
                        records[key] = value#打印损失函数，在records中记录每一个epoch的损失函数,并且记录损失函数大小
                        to_print.append(f'{key}={value:.4f}')
                    print(f'Epoch {epoch}: ' + ', '.join(to_print))

                    scheduler.step()
                    for saver in savers:#根据结果保存结果最好的模型，5折交叉验证，正好保证了所有的valid_dataset都有pred
                        saver.step()


        for saver in savers:
            saver.load()
            test_df_tmp = test_df.copy()
            preds = []
            model_for_train.eval()
            with torch.no_grad():
                with tqdm(test_dl, leave=False, file=sys.stdout) as t:
                    for batch in t:
                        img = batch['img'].to(device)
                        y, _ = model_for_train(img)
                        preds.append(y.cpu().numpy())
            preds=np.concatenate(preds,axis=0)   
            test_df_tmp['pred'] = preds
            results[saver.key].append(test_df_tmp)
        model_for_train.to(cpu)
        model.to(cpu)
        
    for k, v in results.items():
        ensure_path(f'output/{task_path}/results/{k}')
        v = pd.concat(v, axis=0)
        v.to_csv(f'output/{task_path}/results/{k}/data.csv', index=False)

    print("All folds have been excuted")