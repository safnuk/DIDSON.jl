using BSON: @save, @load
using Flux, Flux.Data.MNIST
using Flux: @epochs, onehotbatch, argmax, crossentropy, testmode!, throttle
using Base.Iterators: repeated, partition
# using CuArrays

# Classify MNIST digits with a convolutional network

imgs = MNIST.images()

labels = onehotbatch(MNIST.labels(), 0:9)

# Partition into batches of size 1,000
train = [(cat(4, float.(imgs[i])...), labels[:,i])
         for i in partition(1:60_000, 1000)]

train = gpu.(train)

# Prepare test set (first 1,000 images)
tX = cat(4, float.(MNIST.images(:test)[1:1000])...) |> gpu
tY = onehotbatch(MNIST.labels(:test)[1:1000], 0:9) |> gpu

m = Chain(
    BatchNorm(1; λ=relu),
    Conv((3, 3), 1=>16; pad=(1, 1)),
    BatchNorm(16; λ=relu),
    Conv((3, 3), 16=>16; stride=(2, 2), pad=(1, 1)),
    BatchNorm(16; λ=relu),
    Conv((1, 1), 16=>8),
    x -> maxpool(x, (2,2)),
    # x -> meanpool(x, (28, 28)),
    # x -> mean(x, 1, 2)
    x -> reshape(x, :, size(x, 4)),
    BatchNorm(392; λ=relu),
    Dense(392, 10),
    softmax) |> gpu

testmode!(m)
m(train[1][1])

testmode!(m, false)

loss(x, y) = crossentropy(m(x), y)
function accuracy(x, y; train=true) 
    if !train
        testmode!(m)
    end
    mu = mean(argmax(m(x)) .== argmax(y))
    testmode!(m, false)
    return mu
end

evalcb = throttle(() -> @show(accuracy(tX, tY; train=false)), 120)
opt = ADAM(params(m))

@epochs 10 Flux.train!(loss, train, opt, cb = evalcb)
@save "mymodel10.bson" m

