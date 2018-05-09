from pathlib import Path
import pickle

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


class ImagenetDataset(Dataset):
    def __init__(self, path):
        path = Path(path)
        self.batches = []
        self.labels = []
        self.lengths = []
        for filepath in path.glob("*"):
            x, y = self.load_databatch(filepath)
            self.batches.append(x)
            self.labels.append(y)
            self.lengths.append(x.shape[0])

    def __len__(self):
        return np.sum(self.lengths)

    def __getitem__(self, idx):
        batch = 0
        while idx >= self.lengths[batch]:
            idx -= self.lengths[batch]
            batch += 1
        x = self.batches[batch][idx]
        y = self.labels[batch][idx]
        x = x.astype(np.float32) / 255.0
        x = torch.from_numpy(x)
        return x, y

    def load_databatch(self, filepath, img_size=32):
        with filepath.open('rb') as f:
            d = pickle.load(f)
        x = d['data']
        y = d['labels']

        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i-1 for i in y]

        img_size2 = img_size * img_size

        x = np.concatenate(
            (x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

        return x, y
