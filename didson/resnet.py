import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init

from didson.groupnorm import GroupNorm


def concat_elu(x, inplace=False):
    """
    Like concatenated ReLU (http://arxiv.org/abs/1603.05201),
    but then with ELU
    """
    # axis = list(x.size())
    return F.elu(torch.cat((x, -x), 1), inplace=inplace)


class BasicBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        stride, nonlinearity=F.relu,
        dropout=0.0,
    ):
        super().__init__()
        self.nonlinearity = nonlinearity
        if dropout > 0.0:
            self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        else:
            self.dropout = lambda x: x

        self.alignment = lambda x: x
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        if stride > 1 or in_channels != out_channels:
            self.alignment = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=stride, stride=stride, bias=False
            )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.norm1 = GroupNorm(in_channels)
        self.norm2 = GroupNorm(out_channels)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                weight_init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        original = x
        x = self.nonlinearity(self.norm1(x), inplace=True)
        x = self.conv1(x)
        x = self.nonlinearity(self.norm2(x), inplace=True)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + self.alignment(original)


class BlockStack(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stride=1, blocks=1, Block=BasicBlock, **kwargs):
        super().__init__()
        layers = [Block(in_channels, out_channels, stride, **kwargs)]
        for _ in range(1, blocks):
            layers.append(
                Block(out_channels, out_channels, stride=1, **kwargs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class WideResNet(nn.Module):
    def __init__(self, in_channels, base_channels=16,
                 widening_factor=10, blocks=2,
                 strides=[2, 2, 2], **blockargs):
        super().__init__()
        c = base_channels
        k = widening_factor
        channels = [k * c * 2**n for n in range(len(strides))]
        self.out_channels = channels[-1]
        channels.append(c)
        entry_norm = GroupNorm(in_channels)
        entry = nn.Conv2d(
            in_channels, c, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        layers = [
            BlockStack(channels[n-1], channels[n], stride=strides[n],
                       blocks=blocks, **blockargs)
            for n in range(len(strides))
        ]
        self.layers = nn.Sequential(entry_norm, entry, *layers)

    def forward(self, x):
        return self.layers(x)


class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.resnet = WideResNet(in_channels, **kwargs)
        self.gn = GroupNorm(self.resnet.out_channels)
        self.class_weights = nn.Linear(self.resnet.out_channels, out_channels)

    def forward(self, x):
        x = self.gn(self.resnet(x))
        x = x.mean(2).mean(2)
        return self.class_weights(x)
