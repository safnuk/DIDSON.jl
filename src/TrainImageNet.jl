using BSON: @save
using Flux
using Flux: @epochs, onehotbatch, argmax, logitcrossentropy, testmode!, throttle
using CuArrays

using DIDSON

epochs = 10
width = 10
depth = 16
path = "/home/safnuk/imagenet/samples"
batchsize = 50

function get_data(path, batchsize)
    train = DataLoader(["$path/train_$n.jld" for n in 1:10], batchsize)
    val = DataLoader(["$path/val.jld"], batchsize)
end

m = ResNet(1=>1000; widening_factor=width, depth=depth, base_channels=16, strides=[2, 2, 1]) |> gpu

traindata, testdata = get_data(path, batchsize);

loss(x, y) = logitcrossentropy(m(x), y)
function accuracy(data) 
    testmode!(m)
    mu = 0.0
    n = 0
    for (x, y) in data
        n += 1
        mu += mean(argmax(m(x)) .== argmax(y))
    end
    testmode!(m, false)
    mu / n
end

opt = ADAM(params(m))

for n in 1:epochs
    Flux.train!(loss, traindata, opt)
    @show accuracy(testdata)
    @save "resnet-$(depth)-$(width)_$(n).bson" m
end
