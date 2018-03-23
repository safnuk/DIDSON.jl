using BSON: @save, @load
using Flux, Flux.Data.MNIST
using Flux: @epochs, onehotbatch, argmax, logitcrossentropy, testmode!, throttle
using Base.Iterators: repeated, partition
using CuArrays

# Classify MNIST digits with a convolutional network

imgs = MNIST.images();
test_imgs = MNIST.images(:test);

labels = onehotbatch(MNIST.labels(), 0:9);
test_labels = onehotbatch(MNIST.labels(:test), 0:9);

# Partition into batches of size 100
train = [(cat(4, float.(imgs[i])...), labels[:,i])
         for i in partition(1:60_000, 50)];

train = gpu.(train);

# Prepare test set (first 1,000 images)
testdata = [(cat(4, float.(test_imgs[i])...), test_labels[:,i])
         for i in partition(1:1000, 5)];
testdata = gpu.(testdata);

struct BasicBlock{B, C}
    m::B
    res::C
end

function BasicBlock(channels::Pair{<:Integer,<:Integer}; stride=1, dropout=0.7, σ=relu)
    in, out = channels
    if in == out && stride == 1
        res = identity
    else
        res = Conv((1, 1), in=>out; stride=(stride, stride))
    end
    m = Chain(
        BatchNorm(in; λ=σ),
        Conv((3, 3), in=>out; stride=(stride, stride), pad=(1, 1)),
        BatchNorm(out; λ=σ),
        # Dropout(dropout),
        Conv((3, 3), out=>out; pad=(1, 1))
       )
    BasicBlock(m, res)
end

Flux.treelike(BasicBlock)

function (b::BasicBlock)(x)
    b.res(x) + b.m(x)
end

function WideStack(channels::Pair{<:Integer,<:Integer}; layers=1, stride=1, dropout=0.7, σ=relu)
    in, out = channels
    stack = [BasicBlock(out=>out; dropout=dropout, σ=σ) for n in 1:(layers-1)]
    stack = [BasicBlock(channels; stride=stride, dropout=dropout, σ=σ); stack]
    Chain(stack...)
end

function ResNet(channels::Pair{<:Integer,<:Integer};
                strides=[1, 1, 1], widening_factor=10,
                depth=16, base_channels=16, σ=relu, dropout=0.7)
    in, out = channels
    @assert (depth - 4) % 6 == 0
    n = Int((depth - 4) / 6)
    c = base_channels
    k = widening_factor
    Chain(
        BatchNorm(in),
        Conv((3, 3), in=>c; pad=(1, 1)),
        WideStack(c=>k*c; layers=n, stride=strides[1], dropout=dropout, σ=σ),
        WideStack(k*c=>2k*c; layers=n, stride=strides[2], dropout=dropout, σ=σ),
        WideStack(2k*c=>4k*c; layers=n, stride=strides[3], dropout=dropout, σ=σ),
        BatchNorm(4k*c; λ=relu),
        x -> mean(x, [1, 2]),
        x -> squeeze(x, (1, 2)),
        Dense(4k*c, out))
end

m = ResNet(1=>10; widening_factor=10, depth=16, base_channels=16, strides=[2, 2, 1]) |> gpu


@time m(train[1][1])
@time m(train[1][1])


loss(x, y) = logitcrossentropy(m(x), y)
function accuracy(data; train=true) 
    if !train
        testmode!(m)
    end
    mu = 0.0
    for (x, y) in data
        mu += mean(argmax(m(x)) .== argmax(y))
    end
    testmode!(m, false)
    mu / length(data)
end

evalcb = throttle(() -> @show(accuracy(testdata; train=false)), 60)
opt = ADAM(params(m))

@epochs 10 Flux.train!(loss, train, opt, cb = evalcb)
@save "mymodel10.bson" m

