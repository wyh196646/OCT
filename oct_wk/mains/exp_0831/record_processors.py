from Utils import *


def normal_valid_loss(records):
    losses = np.concatenate(records['normal-valid-loss-list'])
    return 'n-valid-loss', losses.mean()


def abnormal_valid_loss(records):
    losses = np.concatenate(records['abnormal-valid-loss-list'])
    return 'abn-valid-loss', losses.mean()


def train_loss(records):
    losses = reduce(lambda x, y: x + y, records['train-loss-list'])
    return 'train-loss', sum(losses) / len(losses)


def valid_loss(records):
    losses = reduce(lambda x, y: x + y, records['valid-loss-list'])
    return 'valid-loss', sum(losses) / len(losses)


def valid_auc(records, n_class, target_class=-1, record_index=None):
    if record_index is not None:
        pred = torch.cat([x[record_index] for x in records['valid-y_pred-list']]).numpy()
        true = torch.cat([x[record_index] for x in records['valid-y_true-list']]).numpy()
    else:
        pred = torch.cat(records['valid-y_pred-list']).numpy()
        true = torch.cat(records['valid-y_true-list']).numpy()
    try:
        if n_class == 2:
            return f'AUC_{record_index}', roc_auc_score(true, pred[:, 1])
        elif n_class == 1:
            return f'MAE_{record_index}', mean_absolute_error(true, pred)
        else:
            true_oh = np.zeros((true.size, n_class))
            true_oh[np.arange(true.size), true] = 1
            if target_class != -1:
                aucs = roc_auc_score(true_oh, pred, multi_class='ovr', average=None)
                return f'AUC_{record_index}_{target_class}', aucs[target_class]
            else:
                return f'macro_AUC_{record_index}', roc_auc_score(true_oh, pred, multi_class='ovr')
    except Exception as e:
        print(e)
        return f'Metric_{record_index}', float('nan')


def clear_records_epoch(records):
    for key in ['train-y_true-list', 'train-y_pred-list', 'train-loss-list',
                'valid-y_true-list', 'valid-y_pred-list', 'valid-loss-list']:
        records[key] = []


class ModelSaver:
    def __init__(self, model: nn.Module, model_folder, records, key, compare=max, map_location=torch.device('cpu')):
        self.model = model
        self.model_folder = model_folder
        self.records = records
        self.key = key
        self.compare = compare
        self.map_location = map_location
        if self.compare == max:
            self.records[f'best_{key}'] = -float('inf')
        else:
            self.records[f'best_{key}'] = float('inf')

    def step(self):
        if self.compare(self.records[self.key], self.records[f'best_{self.key}']) == self.records[self.key]:
            self.records[f'best_{self.key}'] = self.records[self.key]
            print(f'Save better model, {self.key}={self.records[self.key]:.4f}')
            ensure_path(self.model_folder)
            torch.save(self.model.state_dict(), f'{self.model_folder}/best_{self.key}.pth')

    def load(self):
        self.model.load_state_dict(torch.load(f'{self.model_folder}/best_{self.key}.pth', self.map_location))


class ModelSaverEveryFoldEpoch(ModelSaver):
    def __init__(self, model: nn.Module, model_folder, records, map_location=torch.device('cpu'), fold=0, epoch=0):
        super().__init__(model, model_folder, records, key=f'fold_{fold}-epoch_{epoch}', map_location=map_location)
        self.fold = fold
        self.epoch = epoch
        self.saved = False
    
    def step(self, k):
        if self.records['fold'] == self.fold and self.records['epoch'] == self.epoch:
            self.key = k
            print(f"Save model in epoch {self.records['epoch']}.")
            ensure_path(self.model_folder)
            torch.save(self.model.state_dict(), f"{self.model_folder}/{self.key}.pth")
            self.saved = True
    
    def load(self):
        self.model.load_state_dict(torch.load(f'{self.model_folder}/{self.key}.pth', self.map_location))
