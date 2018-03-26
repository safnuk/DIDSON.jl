from pathlib import Path
import pickle

import numpy as np


def unpickle(filepath):
    with filepath.open('rb') as f:
        d = pickle.load(f)
    return d


def shape_batch(
    idx=None, in_template="train_data_batch_{}",
    out_data="train_data_{}.npy",
    out_labels="train_labels_{}.npy"
):
    img_size = 32
    img_size2 = img_size * img_size
    data = unpickle(Path(in_template.format(idx)))
    x = data['data']
    y = data['labels']
    x = np.concatenate((x[:, :img_size2], x[:, img_size2:2*img_size2],
                        x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], 1, img_size, img_size)).transpose(2, 3, 1, 0)
    y = np.array(y * 3, dtype=np.int)
    np.save(out_data.format(idx), x)
    np.save(out_labels.format(idx), y)


if __name__ == "__main__":
    for idx in range(1, 11):
        shape_batch(idx)

    shape_batch(in_template="val_data", out_data="val_data.npy",
                out_labels="val_labels.npy")
