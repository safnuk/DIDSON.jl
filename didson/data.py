from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset


class Sequence:
    def __init__(self, path, basename):
        path = Path(path)
        self.frames = np.load(
            path / (basename + "_frames.npy")).astype(np.float32)
        self.masks = np.load(
            path / (basename + "_masks.npy")).astype(np.float32)
        self.lengths = np.load(path / (basename + "_lengths.npy"))
        self.centers = np.load(
            path / (basename + "_centers.npy")).astype(np.float32)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        n = self.lengths[idx]
        f = self.frames[idx, :n]
        m = self.masks[idx, :n]
        c = self.centers[idx, :n]
        return (
            torch.from_numpy(f) / 255.0,
            torch.from_numpy(m),
            torch.from_numpy(c))


class SequenceDataset(Dataset):
    # FISH = torch.tensor([0])
    # LAMPREY = torch.tensor([1])
    # OTHER = torch.tensor([2])
    FISH = 0
    LAMPREY = 1
    OTHER = 2

    def __init__(self, path):
        self.fish = Sequence(path, "fish")
        self.lamprey = Sequence(path, "lamprey")
        self.other = Sequence(path, "other")
        self.length = len(self.fish) + len(self.lamprey) + len(self.other)
        self.lamprey_offset = len(self.fish)
        self.other_offset = len(self.fish) + len(self.lamprey)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < self.lamprey_offset:
            return (self.fish[idx], self.FISH)
        elif idx < self.other_offset:
            return (self.lamprey[idx - self.lamprey_offset], self.LAMPREY)
        else:
            return (self.other[idx - self.other_offset], self.OTHER)
