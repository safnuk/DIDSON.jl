__precompile__()
module DIDSON
using PyCall

export DataLoader, samples, view_clip, ResNet
include("Iter.jl")
include("Viewer.jl")
include("ResNet.jl")

function __init__()
end
end # module
