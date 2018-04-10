using BSON
using JLD

function parseObjectIDs(str)
    objects = split(str, ";", keep=false)
    map(parseObject, objects)
end

function parseObject(str)
    ids = split(str, " ", keep=false)
    map(z -> parse(Int, z), ids)
end

function get_object_input(prompt="Enter ';' separated lists of object ids: ")
    print(prompt)
    x = strip(readline())
    parseObjectIDs(x)
end

function save_label(filename, labeldata)
    base, ext = splitext(filename)
    filename = base * ".bson"
    bson(filename, labeldata)
end

function get_next_unlabeled_clip(dir, clip_ext=".avi", done_ext=".done")
    for (root, dirs, files) in walkdir(dir)
        for file in files
            base, ext = splitext(file)
            if ext == clip_ext
                labelfile = joinpath(root, base * done_ext)
                if !isfile(labelfile)
                    return joinpath(root, file)
                end
            end
        end
    end
    return ""
end

function label_next(dir="./", outdir="./")
    used_objects = Vector()
    filename = get_next_unlabeled_clip(dir)
    if filename == ""
        println("No more unlabeled clips")
        return
    end
    println(filename)
    base, ext = splitext(basename(filename))
    clip_data = view_clip(filename)
    lamprey_ids = get_object_input("Enter lamprey object ids: ")
    fish_ids = get_object_input("Enter fish (non-lamprey) object ids: ")
    snippets = extract_snippets(clip_data, lamprey_ids)
    save_snippets(outdir, "lamprey", base, snippets)
    used_objects = [used_objects ; vcat(lamprey_ids...)]
    snippets = extract_snippets(clip_data, fish_ids)
    save_snippets(outdir, "fish", base, snippets)
    used_objects = [used_objects ; vcat(fish_ids...)]
    println("Obj: $(clip_data[:objects])")
    other_ids = [[x.label] for x in clip_data[:objects] if !(x.label in used_objects || is_transient(x))]
    snippets = extract_snippets(clip_data, other_ids)
    save_snippets(outdir, "other", base, snippets)
end

function save_snippets(outdir, class, base, snippets)
    for (counter, snippet) in enumerate(snippets)
        centers, frames = snippet
        mkpath(joinpath(outdir, class))
        path = joinpath(outdir, class, "$(base)_$(counter).jld")
        save(path, "centers", centers, "frames", frames)
    end
end

function extract_snippets(clip_data, object_ids)
    snippets = Vector()
    for x in object_ids
        snippet = extract_snippet(clip_data, x)
        snippets = [snippets ; snippet]
    end
    return snippets
end

function extract_snippet(clip_data, object_ids)
    snippet = Vector()
    objects = [clip_data[:objects][id] for id in object_ids]
    intervals = [keys(x) for x in objects]
    first = min([collect(x)[1] for x in intervals]...)
    last = max([collect(x)[end] for x in intervals]...)
    prev_grabbed = first - 1
    prev_center = [0, 0]
    for frame in first:last
        frame_objects = [object[frame] for object in objects if (frame in keys(object))]
        center = calc_center(frame_objects)
        if prev_grabbed < frame - 1
            extract_interpolated_frames!(snippet, clip_data, prev_grabbed, frame, prev_center, center)
        end
        snippet = push!(snippet, (center, extract_frame(clip_data, center, frame)))
        prev_grabbed = frame
        prev_center = center
    end
    centers = Array{Float64}(2, 0)
    frames = Array{UInt8, 3}(24, 40, 2, 0)
    for snip in snippet
        center, frame = snip
        centers = cat(2, centers, center)
        frames = cat(4, frames, reshape(frame, size(frame)..., 1))
    end
    return (centers, frames)
end

function calc_center(objects)
    total_area = 0.0
    running_x = 0.0
    running_y = 0.0
    for object in objects
        total_area += object.area
        running_x += object.area * object.x.p
        running_y += object.area * object.y.p
    end
    [running_x / total_area, running_y / total_area]
end

function extract_interpolated_frames!(snippet, clip_data, first, last, first_center, last_center)
    Δ = (last_center .- first_center) ./ (last - first)
    for frame in (first+1):(last-1)
        center = first_center + (frame - first) .* Δ 
        push!(snippet, (center, extract_frame(clip_data, center, frame)))
    end
    return snippet
end

function extract_frame(clip_data, center, frame)
    height, width, time = size(clip_data[:clip])
    bbox = calc_bounding_box(center, height, width)
    image = clip_data[:clip][bbox..., frame]
    fg_mask = clip_data[:mask][bbox..., frame]
    cat(3, image, fg_mask)
end

function calc_bounding_box(center, height, width, target_height=24, target_width=40)
    cx, cy = center
    half_height = target_height / 2
    half_width = target_width / 2
    if cx - half_height <= 1
        start_x = 1
    elseif cx + half_height > height
        start_x = height - target_height
    else
        start_x = convert(Int, round(cx - half_height))
    end

    if cy - half_width <= 1
        start_y = 1
    elseif cy + half_width > width
        start_y = width - target_width
    else
        start_y = convert(Int, round(cy - half_width))
    end
    return [start_x:start_x+target_height-1, start_y:start_y+target_width-1]
end

function mark_next_done(dir="./")
    filename = get_next_unlabeled_clip(dir)
    if filename == ""
        println("No more unlabeled clips")
        return
    end
    base, ext = splitext(filename)
    touch(base * ".done")
    println("Finished with $filename")
end
