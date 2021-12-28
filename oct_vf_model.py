import torch
import torch.nn as nn
from torchvision import models


class OCTVFModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.arch = self.config['arch']

        self.loss_fn = nn.MSELoss(reduction='none')

        if self.arch == 'resnet18':
            self.net = models.resnet18(pretrained=True)
            n_out = 512
        elif self.arch == 'resnet50':
            self.net = models.resnet50(pretrained=True)
            n_out = 2048

        self.net.fc = nn.Sequential(
            nn.Linear(n_out, n_out),
            nn.ReLU(),
            nn.Linear(n_out, 52)
        )
    
    def backbone_parameters(self):
        return map(lambda kv: kv[1], filter(lambda kv: not kv[0].startswith('fc.'), self.net.named_parameters()))

    def head_parameters(self):
        return self.net.fc.parameters()

    def _forward(self, img):
        return self.net(img)

    def forward(self, img, label=None, loss_weights=None):
        y = self.net(img)
        loss = None
        raw_loss = None

        if label is not None:
            raw_loss = self.loss_fn(y, label)
            loss = raw_loss * loss_weights
        return y, loss, raw_loss
