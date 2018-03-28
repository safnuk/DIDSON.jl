using Colors, Images, Requires

using BackgroundSegmenter
using ObjectTracker

# minimum area (number of pixels in a time slice) for foreground connected components
const AREA_THRESHOLD = 9
# minimum volume (number of pixels contiguous across time slices)
# for foreground components
const VOLUME_THRESHOLD = 40
const samples = [Pkg.dir("DIDSON") * "/data/lamprey$n.avi" for n in 1:5]

@require ImageView begin
    function view_clip(infile, area=AREA_THRESHOLD, volume=VOLUME_THRESHOLD)
        println("Loading...")
        V = load_video(infile)
        println("Segmenting foreground...")
        @time fgbg = filter(V, area, volume);
        for n in 1:length(V)
            if fgbg[n] == 0
                V[n] = 0
            end
        end
        println("Collecting blobs...")
        objects = Vector{Object}()
        num_objects = 0
        @time blob_series = form_blobs(fgbg[:, :, 2:end])

        println("Loading video player...")
        gui = imshow(V)
        println("Tracking objects...")
        for (n, blobs) in enumerate(blob_series)
            num_objects = match!(objects, blobs, num_objects; radius=20)
            for obj in objects
                # if is_transient(obj)
                #     color = RGB(1, 0, 0)
                # else
                if !is_transient(obj)
                    color = RGB(0, 0, 1)
                    annotate!(gui, AnnotationPoint(obj.y.p, obj.x.p, z=n+1, shape='.', size=2, color=color))
                    annotate!(gui, AnnotationText(obj.y.p,
                                                obj.x.p+4,
                                                z=n+1, string(obj.label),
                                                color=color, fontsize=4))
                end
            end
        end
        println("Done")
        return
    end
end

function load_video(infile)
    ext = infile[end-3:end]
    if ext == ".avi"
        return reinterpret(UInt8, load(infile))[1, :, :, :] 
    else
        throw(ArgumentError("invalid file type - must be avi"))
    end
end

function filter(V, min_area, min_volume)
    fgbg = zeros(V);
    (n, m, t) = size(V)
    M = MixtureModel(n, m);
    for i in 1:t
        apply!(M, (view(V, :, :, i)), (view(fgbg, :, :, i)));
    end
    if min_area > 0
        for i in 1:t
            c = view(fgbg, :, :, i);
            # filter_components!(temp1, c, min_area);
            fgbg[:, :, i] = filter_components(c, min_area);
            fgbg[:, :, i] = morphological_close(c)
        end
    end
    fgbg = filter_components(fgbg, min_volume);
    return fgbg
end
