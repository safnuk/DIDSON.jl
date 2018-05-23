using JLD

const MAX_FRAMES = 128
const STRIDE = 64
indir = "./"
outdir = "../labeled_test$MAX_FRAMES"


function splitdata(data, maxframes=MAX_FRAMES, stride=STRIDE)
    splits = []
    c = data["centers"]
    f = data["frames"]
    n = size(c, 2)
    pos = 1
    while pos + maxframes <= n
        range = pos:(pos + maxframes - 1)
        push!(splits, Dict("centers" => c[:, range], "frames" => f[:, :, :, range]))
        pos += stride
    end
    range = max(1, n-maxframes+1):n
    push!(splits, Dict("centers" => c[:, range], "frames" => f[:, :, :, range]))
    return splits
end

for (root, dirs, files) in walkdir(indir)
    for file in files
        outpath = joinpath(outdir, root)
        inpath = joinpath(root, file)
        data = load(inpath)
        splits = splitdata(data)
        for (n, split) in enumerate(splits)
            base, ext = splitext(file)
            filename = base * "_$n" * ext
            save(joinpath(outpath, filename), split)
        end
    end
end
