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

struct BasicBlock{B, C}
    m::B
    res::C
end

function BasicBlock(channels::Pair{<:Integer,<:Integer}; stride=1, dropout=0.7, σ=relu)
    in, out = channels
    if in == out
        res = identity
    else
        res = Conv((1, 1), in=>out; stride=(stride, stride))
    end
    m = Chain(
        BatchNorm(in; λ=σ),
        Conv((3, 3), in=>out; stride=(stride, stride), pad=(1, 1)),
        BatchNorm(out; λ=σ),
        Dropout(dropout),
        Conv((3, 3), out=>out; pad=(1, 1))
       )
    BasicBlock(m, res)
end

Flux.treelike(BasicBlock)

function (b::BasicBlock)(x)
    b.res(x) + b.m(x)
end

p = 1.0
m = Chain(
    BasicBlock(1=>8; stride=2, dropout=p),
    BasicBlock(8=>8, dropout=p),
    BasicBlock(8=>16; stride=2, dropout=p),
    BasicBlock(16=>16; dropout=p),
    BatchNorm(16; λ=relu),
    x -> mean(x, [1, 2]),
    x -> reshape(x, 16, :),
    Dense(16, 10),
    softmax
   ) |> gpu

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

