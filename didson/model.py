from pathlib import Path

import torch

from modelmanager import Model

from didson.data import ImagenetDataset
from didson.resnet import Classifier


class Accuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, targets):
        max_index = x.max(dim=1)[1]
        return (max_index == targets).float().mean()


class DidsonModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_size = 4
        self.metrics["accuracy"] = Accuracy()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loggers = {}

    def load_data(self, basepath):
        basepath = Path(basepath)
        train = basepath / "train"
        validation = basepath / "val"
        self.train_data = ImagenetDataset(train)
        if validation.exists():
            self.validation_data = ImagenetDataset(validation)
        else:
            self.validation_data = None

    def init_network(self, **network):
        input, _ = self.train_data[0]
        network['in_channels'] = input.shape[0]
        self.net = Classifier(**network)

    def get_visuals(self):
        return {}
