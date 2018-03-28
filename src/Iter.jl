using Base.Iterators: partition
using Flux
using Flux: onehotbatch
using JLD

mutable struct DataLoader{T, F}
    files::Vector{String}
    buffer::T
    batchsize::Int
    transform::F
    current_file_idx::Int
end

function DataLoader(files, batchsize; transform=((x, y)->(x./255.0f0, onehotbatch(y, 1:1000))))
    DataLoader(files, load_buffer(files[1], batchsize), batchsize, transform, 1)
end

function load_buffer(filename, batchsize)
    println("Loading $filename")
    d = load(filename)
    x, y = (d["data"], d["labels"])
    n = size(x)[end]
    @assert n == size(y)[end]
    [(x[:, :, :, i], y[i]) for i in partition(1:n, batchsize)]
end

Base.start(::DataLoader) = 1
function Base.next(dl::DataLoader, state)
    data, state = next(dl.buffer, state)
    if done(dl.buffer, state) && dl.current_file_idx < length(dl.files)
        dl.current_file_idx += 1
        dl.buffer = load_buffer(dl.files[dl.current_file_idx], dl.batchsize)
        state = start(dl.buffer)
    end
    data = gpu.(dl.transform(data...))
    [data], state
end

Base.done(dl::DataLoader, state) = done(dl.buffer, state)
