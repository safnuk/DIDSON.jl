from pathlib import Path

import torch

from modelmanager import Model

from didson.resnet import WideResNet, Classifier
import cae.cfd.dataloader as cfd_dl


networks = {
    'WideResNet': WideResNet,
    'Classifier': Classifier,
}


class Accuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, targets):
        max_index = x.max(dim=1)[1]
        return (max_index == targets).sum()


class Model(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_size = 4
        self.metrics["accuracy"] = Accuracy()

    def load_data(self, basepath):
        basepath = Path(basepath)
        train = basepath / "train"
        validation = basepath / "eval"
        self.train_data = cfd_dl.DirectoryDataset(train)
        if validation.exists():
            self.validation_data = cfd_dl.DirectoryDataset(validation)
        else:
            self.validation_data = None

    def init_network(self, id, **network):
        input, _ = self.train_data[0]
        network['in_channels'] = input.shape[0]
        module = networks[id]
        self.net = module(**network)

    def get_visuals(self):
        return {}
