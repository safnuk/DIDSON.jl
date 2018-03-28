using PyCall
using JLD

@pyimport numpy as np

function convert_to_jld()
    for idx in 1:10
        x = np.load("train_data_$idx.npy");
        y = np.load("train_labels_$idx.npy");
        save("train_$idx.jld", "data", x, "labels", y)
    end


    x = np.load("val_data.npy");
    y = np.load("val_labels.npy");
    save("val.jld", "data", x, "labels", y)
end

function create_sample_data()
    for idx in 1:10
        data = load("train_$idx.jld")
        x, y = data["data"], data["labels"]
        x = x[:, :, :, 1:100]
        y = y[1:100]
        save("samples/train_$idx.jld", "data", x, "labels", y)
    end

    data = load("val.jld")
    x, y = data["data"], data["labels"]
    x = x[:, :, :, 1:200]
    y = y[1:200]
    save("samples/val.jld", "data", x, "labels", y)
end

create_sample_data()
