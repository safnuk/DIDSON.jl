using PyCall
using JLD

@pyimport numpy as np

for idx in 1:10
    x = np.load("train_data_$idx.npy");
    y = np.load("train_labels_$idx.npy");
    save("train_$idx.jld", "data", x, "labels", y)
end


x = np.load("val_data.npy");
y = np.load("val_labels.npy");
save("val.jld", "data", x, "labels", y)
