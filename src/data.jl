using FileIO
using ImageIO
using Images: channelview, RGB
using LinearAlgebra
using ArgCheck
using Random
using JSON3
using JLD2
using ProgressLogging
using ResumableFunctions

using JSON3.StructTypes: StructType, Struct

const Vec3 = NTuple{3,Float32}


Base.@kwdef struct CameraView
    camera_direction::Vec3
    camera_origin::Vec3
    x_axis::Vec3
    y_axis::Vec3
    x_fov::Float32
    y_fov::Float32
end

function CameraView(path)
    data = open(path, "r") do io
        JSON3.read(io)
    end
    CameraView(
        camera_direction=Vec3(data["z"]),
        camera_origin=Vec3(data["origin"]),
        x_axis=Vec3(data["x"]),
        y_axis=Vec3(data["y"]),
        x_fov=Float32(data["x_fov"]),
        y_fov=Float32(data["y_fov"])
    )
end

"""
Get all of the rays in the view in raster scan order

Returns 3x2xB array of (origin, direction) pairs
"""
function bare_rays(cv::CameraView, width::Int, height::Int)
    z = collect(cv.camera_direction)
    @check size(z) == (3,)
    rgh = collect(LinRange(-1, 1, height))
    rgh = reshape(rgh, 1, 1, size(rgh)...)
    ys = tan(cv.y_fov / 2) .* rgh .* collect(cv.y_axis)
    @check size(ys) == (3, 1, height)
    rgw = collect(LinRange(-1, 1, width))
    rgw = reshape(rgw, 1, size(rgw)..., 1)
    xs = tan(cv.x_fov / 2) .* rgw .* collect(cv.x_axis)
    @check size(xs) == (3, width, 1)
    directions = reshape(xs .+ ys .+ z, 3, 1, :)
    directions ./= map(norm, eachslice(directions; dims=1))
    co = collect(cv.camera_origin)
    co = reshape(co, size(co)..., 1, 1)
    origins = reshape(repeat(co, 1, width, height), 3, 1, :)
    @check size(origins) == size(directions) == (3, 1, height*width)
    return Float32.(hcat(origins, directions))
end

struct FileNeRF
    imagepath::String
    cameraview::CameraView
end

function FileNeRF(imagepath)
    camerapath = first(splitext(imagepath)) * ".json"
    cv = CameraView(camerapath)
    return FileNeRF(imagepath, cv)
end

function image(fn::FileNeRF)
    return load(fn.imagepath)
end

function rays(fn::FileNeRF)
    # https://discourse.julialang.org/t/reading-png-rgb-channels-julia-vs-python/73599/12
    # Note: this is not the same as PIL.Image something about linear vs SRGB
    img = channelview(RGB.(image(fn)))
    bare = bare_rays(fn.cameraview, size(img)[begin+1:end]...) # TODO
    colors = Float32.(reshape(img, 3, 1, :))
    return hcat(bare, colors)
end

struct ModelMetadata
    bbox_min::Vec3
    bbox_max::Vec3
end

function ModelMetadata(path::String)
    metadata = open(path, "r") do io
        JSON3.read(io)
    end
    ModelMetadata(Vec3(metadata["min"]), Vec3(metadata["max"]))
end

struct NeRFDataset
    metadata::ModelMetadata
    views::Vector{FileNeRF}
end

function NeRFDataset(path::AbstractString)
    metadata = ModelMetadata(joinpath(path, "metadata.json"))
    img_paths = filter!(endswith("png"), readdir(path; join=true))
    views = map(FileNeRF, img_paths)
    return NeRFDataset(metadata, views)
end

Base.length(d::NeRFDataset) = length(d.views)
Base.getindex(d::NeRFDataset, idx::Integer) = rays(d.views[idx])

function iterate_batches(ds::NeRFDataset, savepath::AbstractString; rng::AbstractRNG=Random.GLOBAL_RNG, num_shards::Int=32, batch_size::Int=512, repeat::Bool=true)
    sd = ShardedNeRFDataset(ds, savepath; rng, num_shards)
    return iterate_batches(sd; batch_size, rng, repeat)
end

struct ShardedNeRFDataset{S<:AbstractString}
    files::Vector{S}
end

function ShardedNeRFDataset(path::AbstractString)
    files = filter!(endswith("jld2"), readdir(path; join=true))
    @check length(files) > 0
    return ShardedNeRFDataset(files)
end

Base.length(ds::ShardedNeRFDataset) = length(ds.files)

function ShardedNeRFDataset(ds::NeRFDataset, savepath::AbstractString; rng::AbstractRNG=Random.GLOBAL_RNG, num_shards::Int=32)
    if !isdir(savepath)
        mkpath(savepath)
    else
        files = filter!(endswith("jld2"), readdir(savepath; join=true))
        if length(files) == num_shards
            return ShardedNeRFDataset(files)
        end
    end
    shards = Vector{Array{Float32,3}}(undef, num_shards)
    @progress for view in ds.views
        rs = rays(view)
        assignments = rand(rng, 1:num_shards, (size(rs)[end],))
        for sid in 1:num_shards
            sub_batch = rs[:, :, assignments .== sid]
            if size(sub_batch, 3) > 0
                try
                    shards[sid] = cat(shards[sid], sub_batch; dims=3)
                catch e
                    if e isa UndefRefError
                        shards[sid] = sub_batch
                    end
                end
            end
        end
    end
    files = [joinpath(savepath, "$i.jld2") for i in 1:num_shards]
    for i in 1:num_shards
        jldopen(files[i], "w") do f
            f["rays"] = shards[i]
        end
    end
    return ShardedNeRFDataset(files)
end

@resumable function iterate_batches(ds::ShardedNeRFDataset; batch_size::Int, rng::AbstractRNG=Random.GLOBAL_RNG, repeat::Bool=false)
    shard_indices = randperm(rng, length(ds))
    current_batch = nothing
    while true
        for shard in shard_indices
            shard_rays = JLD2.load(ds.files[shard], "rays")
            if current_batch === nothing
                current_batch = shard_rays
            else
                current_batch = cat(current_batch, shard_rays; dims=3)
            end
            while size(current_batch)[end] >= batch_size
                @yield current_batch[:, :, 1:batch_size]
                current_batch = current_batch[:, :, batch_size+1:end]
            end
        end
        if !repeat
            break
        end
    end
    if size(current_batch)[end] > 0
        @yield current_batch
    end
end