using BSON: @load
using JLD
using Flux

feature_model_file = "resnet-gn-16-10-best.bson"
out_features = 12*20*640
N = 15

function load_data(basedir)
    data = []
    lamprey = ([1.0, 0.0, 0.0], "lamprey")
    fish = ([0.0, 1.0, 0.0], "fish")
    other = ([0.0, 0.0, 1.0], "other")
    for category in [lamprey, fish, other]
        label, dir = category
        path = joinpath(basedir, dir) 
        for (root, dirs, files) in walkdir(path)
            for file in files
                filepath = joinpath(root, file)
                x = load(filepath)
                x["frames"] = x["frames"] ./ 1.0
                x["frames"][:, :, 1, :] = x["frames"][:, :, 1, :] ./ 255.0
                y = copy(label)
                push!(data, (x, y))
            end
        end
    end
    return data
end

# @load feature_model_file feature_model

feature_model = Chain(
    Conv((3,3), 1=>640, relu; pad=(1, 1)),
    x -> maxpool(x, (2,2)),
    x -> mean(x, [1, 2]),
    x -> squeeze(x, (1, 2)),
    Dense(640, 1)
)

feature_layer = feature_model[1:end-3] |> gpu


downsample = Dense(out_features+2, N, σ) |> gpu
scanner = LSTM(N, N) |> gpu
predictor = Dense(N, 3) |> gpu

function model(x)
    centers = x["centers"] |> gpu
    frames = x["frames"][:, :, 1:1, :] |> gpu

    features = feature_layer(frames)
    features = reshape(features, out_features, :)
    inputs = downsample(vcat(features, centers))
    n, T = size(inputs)
    state = scanner(inputs[:, 1])
    for j in 2:T
        state = scanner(inputs[:, j])
    end
    reset!(scanner)
    softmax(predictor(state))
end

