import BSON
import JLD
using Flux
using Flux: back!, crossentropy

feature_model_file = "resnet-gn-16-10-best.bson"
trainpath = "labeled"
testpath = "labeled_test"
out_features = 12*20*640
N = 10

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
                x = JLD.load(filepath)
                x["frames"] = x["frames"] ./ 1.0
                x["frames"][:, :, 1, :] = x["frames"][:, :, 1, :] ./ 255.0
                y = copy(label)
                push!(data, (x, y))
            end
        end
    end
    return data
end

# BSON.@load feature_model_file feature_model

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

function accuracy(data)
    mu = 0.0
    n = 0
    for (x, y) in data
        n += 1
        out = model(x)
        y = gpu(y)
        loss += crossentropy(out, y) 
        mu += mean(argmax(out) .== argmax(y))
    end
    loss / n, mu / n
end

traindata = load_data(trainpath)
testdata = load_data(testpath)
loss(x, y) = crossentropy(model(x), y)
opt = ADAM(params(downsample, scanner, predictor))

macro interrupts(ex)
  :(try $(esc(ex))
    catch e
      e isa InterruptException || rethrow()
      throw(e)
    end)
end

function train(loss, data, opt, freq=10)
    loss_tracker = []
    for (n, d) in enumerate(data)
        l = loss(d...)
        push!(loss_tracker, l)
        isinf(l) && error("Loss is Inf")
        isnan(l) && error("Loss is NaN")
        @interrupts back!(l)
        opt()
        if n % freq == 0
            avg_loss = mean(loss_tracker[end-freq+1:end])
            @show avg_loss
        end
    end
    println("Epoch average loss: $(mean(loss_tracker))")
end

function run_rnn(loss, opt, traindata, testdata, end_epoch, start_epoch=1)
    for n in start_epoch:end_epoch
        println("Epoch $n ==============")
        train(loss, traindata, opt)
        @show accuracy(testdata)
        BSON.@save "rnn_models.bson" downsample scanner predictor
    end
end
