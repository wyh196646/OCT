import torch
import torch.nn as nn
from torchvision import models


class MultiTaskHead(nn.Module):
    def __init__(self, n_classes, in_size=2048):
        super(MultiTaskHead, self).__init__()
        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(in_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_class)
        ) for n_class in n_classes])

    def forward(self, x):
        return [fc(x) for fc in self.fcs], x


class MultiTaskLoss(nn.Module):
    def __init__(self, n_classes):
        super(MultiTaskLoss, self).__init__()
        self.n_classes = n_classes
        self.loss_fns = []
        for n_class in self.n_classes:
            if n_class != 1:
                self.loss_fns.append(nn.CrossEntropyLoss(reduction='none'))
            else:
                self.loss_fns.append(nn.MSELoss(reduction='none'))

    def forward(self, preds, labels):
        for i in range(len(preds)):
            if self.n_classes[i] != 1:
                labels[i] = labels[i].long()
        loss = torch.cat([self.loss_fns[i](preds[i], labels[i]) for i in range(len(preds))])
        loss = loss.mean()
        return loss


class OCTVFModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predict = False
        self.config = config
        self.arch = self.config['arch']
        self.n_classes = self.config['n_classes']

        if self.arch == 'resnet18':
            self.net = models.resnet18(pretrained=True)
            n_out = 512
        elif self.arch == 'resnet50':
            self.net = models.resnet50(pretrained=True)
            n_out = 2048

        self.net.fc = MultiTaskHead(self.n_classes, in_size=n_out)
    
    def backbone_parameters(self):
        return map(lambda kv: kv[1], filter(lambda kv: not kv[0].startswith('fc.'), self.net.named_parameters()))

    def head_parameters(self):
        return self.net.fc.parameters()

    def forward(self, img):
        pred, embedding = self.net(img)
        if self.predict:
            for i in range(len(pred)):
                if self.n_classes[i] != 1:
                    pred[i] = torch.softmax(pred[i], dim=-1)
            return pred, embedding
        else:
            for i in range(len(pred)):
                if self.n_classes[i] == 1:
                    pred[i] = pred[i].squeeze(-1)
            return pred, embedding
