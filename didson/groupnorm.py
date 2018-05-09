import torch
import torch.nn as nn


class GroupNorm(nn.Module):
    """
    Implementation of group normalization from
    https://arxiv.org/abs/1803.08494
    Authors find a large drop in performance if normalization is
    performed across each channels separately (i.e. num_groups = channels),
    which coincides with instance normalization.
    """
    def __init__(self, channels, channels_per_group=8, eps=1e-2):
        self.always_float = True
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        if channels <= channels_per_group:
            self.num_groups = 1
        else:
            assert channels % channels_per_group == 0
            self.num_groups = channels // channels_per_group
        self.eps = eps

    def forward(self, x):
        if type(x.data) == torch.cuda.HalfTensor:
            half = True
            x = x.float()
        else:
            half = False
        original_size = x.size()
        N = original_size[0]
        C = original_size[1]
        G = self.num_groups
        template = [1] * len(original_size)
        template[1] = C
        W = self.weight.view(template)
        b = self.bias.view(template)

        x = x.view(N, G, -1)
        mean = x.float().mean(-1, keepdim=True)
        var = x.float().var(-1, keepdim=True)

        x = ((x-mean) / (var+self.eps)).sqrt()
        x = x.view(original_size)
        x = x * W + b
        if half:
            return x.half()
        else:
            return x
