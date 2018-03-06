using PyCall
using ImageView, GtkReactive, Colors, Images

using BackgroundSegmenter
using ObjectTracker

@pyimport skvideo as skv
@pyimport numpy as np
@pyimport skvideo.io as skvio

function load(infile, limit=0)
    V = np.load(infile)
    (t, n, m) = size(V)
    if limit == 0
        return V
    end
    left = convert(Int, round((n - limit) / 2))
    top = convert(Int, round((m - limit) / 2))
    return V[:, left:(left+limit), top:(top+limit)]
end

function time_results(V, fgbg, M, t)
    @time for i in 2:t
        apply!(M, (view(V, i, :, :)), (view(fgbg, i, :, :)));
    end
end

function filter(V, min_area=8, min_volume=40)
    fgbg = zeros(V);
    (t, n, m) = size(V)
    M = MixtureModel(n, m);
    for i in 1:t
        apply!(M, (view(V, i, :, :)), (view(fgbg, i, :, :)));
    end
    if min_area > 0
        for i in 1:t
            c = view(fgbg, i, :, :);
            fgbg[i, :, :] = filter_components(c, min_area);
            fgbg[i, :, :] = morphological_close(c)
        end
    end
    if min_volume > 0
        fgbg = filter_components(fgbg, min_volume);
    end
    return fgbg
end

f1 = BackgroundSegmenter.Filter(1, 0.2);
f2 = BackgroundSegmenter.Filter(1, 0.4);
# f2 = BackgroundSegmenter.Filter(3, 0.08);
# f3 = BackgroundSegmenter.Filter(5, 0.04);
V1 = load("data/lamprey1.npy");
V2 = load("data/lamprey2.npy");
V2 = V2[:, 300:450, 150:400];
V = skvio.vread("data/lamprey3.avi")[:, :, :, 3];
V = skvio.vread("data/lamprey4.avi")[:, :, :, 3];
V = skvio.vread("data/lamprey5.avi")[:, :, :, 3];
V = skvio.vread("data/lamprey6.avi")[:, :, :, 3];

V = V2;

fgbg = filter(V, 8, 20);
gui = imshow(permutedims(fgbg, [2, 3, 1]))
# gui = imshow(permutedims(fgbg, [2, 3, 1]))
# fgbg1 = apply(f, fgbg);
# A0 = permutedims(V, [2, 3, 1]);
# A1 = permutedims(scale(V, fgbg1), [2, 3, 1]);
# imshow(hcat(A0, A1))

objects = Vector{Object}()
num_objects = 0
@time blob_series = form_blobs(fgbg[2:end, :, :])
@time for (n, blobs) in enumerate(blob_series)
    num_objects = match!(objects, blobs, num_objects; radius=20)
    for obj in objects
        if is_transient(obj)
            color = RGB(1, 0, 0)
        else
        # if !is_transient(obj)
            color = RGB(0, 0, 1)
        end
        annotate!(gui, AnnotationPoint(obj.centroid[2], obj.centroid[1], z=n+1, shape='.', size=2, color=color))
        annotate!(gui, AnnotationText(obj.centroid[2],
                                      obj.centroid[1]+4,
                                      z=n+1, string(obj.label),
                                      color=color, fontsize=3))
        if !is_transient(obj)
            annotate!(gui, AnnotationBox(obj.centroid[2] - 20,
                                        obj.centroid[1] - 8,
                                        obj.centroid[2] + 20,
                                        obj.centroid[1] + 8,
                                        z=n+1, color=RGB(0, 1, 0)))
        end
    end
end

# W = @view V[:, :, 2:end];
# colors = distinguishable_colors(blobs+1);
# components = Array{RGB{N0f8}}(size(fgbg));
# for idx in eachindex(fgbg)
#     components[idx] = colors[fgbg[idx]+1] * (W[idx] * 1.0/255.0);
# end
# imshow(components)
# @time fgbg2 = apply(f2, fgbg);
# @time fgbg3 = apply(f3, fgbg);
# A2 = scale(V, fgbg2);
# A3 = scale(V, fgbg3);

# C = zeros(UInt8, 2n, 2m, t);
# C[1:n, 1:m, :] = A0;
# C[n+1:end, 1:m, :] = A2;
# C[1:n, m+1:end, :] = A1;
# C[n+1:end, m+1:end, :] = A3;

# idx = annotate!(guidict, AnnotationBox((20, 30), (40, 20), linewidth=2, color=RGB(0, 1, 0)))
# mm =  [MixtureModel(5) for i in 1:n, j in 1:m];
# cut = MinCut(n, m);
# for i in 1:5
#     time_results(V, fgbg, cut,  mm, i)
# end
# @time  for i in 6:t-1
#     fgbg[i, :, :] = label_components(apply_mrf!(cut, mm, (@view V[i, :, :]), 2.0), 12)
# end
# time_results(V, fgbg, cut,  mm, t)

# A = permutedims(V .* fgbg, [2, 3, 1])
