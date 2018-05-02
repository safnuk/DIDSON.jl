using JLD

const MAX_FRAMES = 20
const MAX_OVERLAP = 15 
indir = "./"
outdir = "../labeled_trunc"


function splitdata(data, maxframes=MAX_FRAMES, maxoverlap=MAX_OVERLAP)
    splits = []
    c = data["centers"]
    f = data["frames"]
    n = length(c)
    num_splits = div(n, maxframes)
    if n % maxframes < (maxframes - maxoverlap)
        num_splits -= 1
    end
    if n <= maxframes 
        push!(splits, Dict("centers" => c, "frames" => f))
    else
        push!(splits, Dict("centers" => c[1:maxframes], "frames" => f[1:maxframes]))
    end
    for m in 1:num_splits-1
        range = (m * maxframes + 1) : ((m+1) * maxframes)
        push!(splits, Dict("centers" => c[range], "frames" => f[range]))
    end
    if num_splits >= 1
        range = (n - maxframes + 1) : n
        push!(splits, Dict("centers" => c[range], "frames" => f[range]))
    end
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
