using BSON: @save
using Flux
using Flux: @epochs, onehotbatch, argmax, logitcrossentropy, testmode!, throttle
using CuArrays

using DIDSON

epochs = 4
width = 10
depth = 16
path = "$(homedir())/imagenet/samples"
batchsize = 50

function get_data(path, batchsize)
    train = DataLoader(["$path/train_$n.jld" for n in 1:10], batchsize)
    val = DataLoader(["$path/val.jld"], batchsize)
    (train, val)
end

function run(loss, opt, traindata, testdata, epochs, start_epoch=1)
    for n in start_epoch:start_epoch + epochs
        Flux.train!(loss, traindata, opt)
        @show accuracy(testdata)
        @save "resnet-gn-$(depth)-$(width)_$(n).bson" m
    end
end

function accuracy(data) 
    mu = 0.0
    n = 0
    for (x, y) in data
        n += 1
        mu += mean(argmax(m(x)) .== argmax(y))
    end
    mu / n
end

m = ResNet(1=>1000; widening_factor=width, depth=depth, base_channels=16, strides=[1, 1, 1]) |> gpu
traindata, testdata = get_data(path, batchsize);
loss(x, y) = logitcrossentropy(m(x), y)
opt = ADAM(params(m))

# run(loss, opt, traindata, testdata, epochs)
