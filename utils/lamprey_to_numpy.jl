using PyCall
using JLD

@pyimport numpy as np

function convert_to_numpy(path)
    for (root, dirs, files) in walkdir(path)
        frame_stack = Array{UInt8}(24, 40, 1, 20, 0)
        mask_stack = Array{UInt8}(24, 40, 1, 20, 0)
        length_stack = Array{Int}(0)
        center_stack = Array{Float64}(2, 20, 0)
        for file in files
            inpath = joinpath(root, file)
            data = load(inpath)
            centers = data["centers"]
            f = data["frames"]
            frames = f[:, :, 1:1, :]
            masks = f[:, :, 2:2, :]
            n = size(frames, 4)
            T = eltype(frames)
            S = eltype(centers)
            c_pad = zeros(S, 2, 20 - n)
            f_pad = zeros(T, 24, 40, 1, 20 - n)
            centers = reshape(cat(2, centers, c_pad), 2, 20, 1)
            frames = reshape(cat(4, frames, f_pad), 24, 40, 1, 20, 1)
            masks = reshape(cat(4, masks, f_pad), 24, 40, 1, 20, 1)
            center_stack = cat(3, center_stack, centers)
            frame_stack = cat(5, frame_stack, frames)
            mask_stack = cat(5, mask_stack, masks)
            length_stack = vcat(length_stack, n)
        end
        center_stack = PyReverseDims(center_stack)
        frame_stack = PyReverseDims(frame_stack)
        mask_stack = PyReverseDims(mask_stack)
        if length(length_stack) > 0
            np.save(root * "_centers.npy", center_stack)
            np.save(root * "_frames.npy", frame_stack)
            np.save(root * "_masks.npy", mask_stack)
            np.save(root * "_lengths.npy", length_stack)
        end
    end
end

convert_to_numpy("./")
