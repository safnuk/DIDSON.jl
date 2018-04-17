using Colors, Images, Requires
using DataStructures
# using ImageView

using BackgroundSegmenter
using ObjectTracker

# minimum area (number of pixels in a time slice) for foreground connected components
const AREA_THRESHOLD = 9
# minimum volume (number of pixels contiguous across time slices)
# for foreground components
const VOLUME_THRESHOLD = 40
const samples = [Pkg.dir("DIDSON") * "/data/lamprey$n.avi" for n in 1:5]

function view_clip(infile, area=AREA_THRESHOLD, volume=VOLUME_THRESHOLD)
    V = load_video(infile)
    height, width, time = size(V)
    W = zeros(V)
    fgbg = filter(V);
    for n in 1:length(V)
        if fgbg[n] != 0
            W[n] = V[n]
        end
    end
    object_history = DefaultDict{Int, OrderedDict{Int, Object}}(OrderedDict{Int, Object}) 
    objects = Vector{Object}()
    num_objects = 0
    blob_series = form_blobs(fgbg[:, :, 2:end])

    println("Loading video player...")
    gui = imshow(hcat(W, V))
    println("Tracking objects...")
    for (n, blobs) in enumerate(blob_series)
        num_objects = match!(objects, blobs, num_objects; radius=20)
        for obj in objects
            object_history[obj.label][n+1] = obj
        end
        for obj in objects
            if is_transient(obj)
                color = RGB(1, 0, 0)
            else
            # if !is_transient(obj)
                color = RGB(0, 0, 1)
            end
            annotate!(gui, AnnotationPoint(obj.y.p, obj.x.p, z=n+1, shape='.', size=2, color=color))
            annotate!(gui, AnnotationPoint(obj.y.p + width, obj.x.p, z=n+1, shape='.', size=2, color=color))
            annotate!(gui, AnnotationText(obj.y.p,
                                        obj.x.p+4,
                                        z=n+1, string(obj.label),
                                        color=color, fontsize=4))
        end
    end
    return Dict(:objects => object_history, :clip => V, :mask => fgbg)
end

function load_video(infile)
    ext = infile[end-3:end]
    if ext == ".avi"
        return reinterpret(UInt8, load(infile))[1, :, :, :] 
    else
        throw(ArgumentError("invalid file type - must be avi"))
    end
end

function filter(V; radius=5, min_neighbors=20, min_cluster_size=30)
    fgbg = zeros(V);
    (n, m, t) = size(V)
    M = MixtureModel(n, m);
    for i in 1:t
        apply!(M, (view(V, :, :, i)), (view(fgbg, :, :, i)));
    end
    fgbg[:, :, 1] = zeros(fgbg[:, :, 1])
    fgbg = cluster(fgbg, radius; min_neighbors=min_neighbors, min_cluster_size=min_cluster_size)
    for i in 1:t
        c = view(fgbg, :, :, i);
        fgbg[:, :, i] = morphological_close(c)
    end
    return fgbg
end
