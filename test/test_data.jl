using SimpleNeRF: NeRFDataset, CameraView, FileNeRF, ModelMetadata, iterate_batches, RGB, channelview
using Random
using Statistics
using LinearAlgebra
using Test

cv1 = CameraView(camera_direction=(0.0, 1.0, 0.0),
    camera_origin=(2.0, 2.0, 2.0),
    x_axis=(-1.0, 0.0, 0.0),
    y_axis=(0.0, 0.0, 1.0),
    x_fov=60.0,
    y_fov=60.0,)

cv2 = CameraView(camera_direction=(1.0, 0.0, 0.0),
    camera_origin=(-2.0, 2.0, 2.0),
    x_axis=(-0.0, 0.0, -1.0),
    y_axis=(0.0, 1.0, 0.0),
    x_fov=60.0,
    y_fov=60.0,)

file1 = FileNeRF(joinpath(@__DIR__, "testimg1.png"), cv1)
file2 = FileNeRF(joinpath(@__DIR__, "testimg2.png"), cv2)
ds = NeRFDataset(ModelMetadata((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), [file1, file2])
batches = mktempdir() do tmp_dir
    [d for d in iterate_batches(ds, tmp_dir; rng=MersenneTwister(42), batch_size=51, repeat=false)]
end

@test length(batches) == 4
@test size(batches[end])[end] == 200 - 51*3

combined = cat(batches...; dims=3)

for view in ds.views
    # samples w.r.t origin
    origin = collect(view.cameraview.camera_origin)
    origins = combined[:, 1, :]
    view_mask = sum(abs, origins .- origin; dims=1) .< 1e-5
    count = sum(view_mask)
    num_pixels = prod(size(SimpleNeRF.image(view)))
    @test round(Int, count) == num_pixels

    view_rays = combined[:, :, vec(view_mask)]
    
    # camera direction
    directions = view_rays[:, 2, :]
    mean_direction = mean(directions; dims=ndims(directions))
    mean_direction /= norm(mean_direction)
    camera_dot = sum(mean_direction .* collect(view.cameraview.camera_direction))
    @test abs(Float32(camera_dot) - 1) < 1e-5

    # color validity
    colors = view_rays[:, 3, :]
    mean_color = mean(colors; dims=ndims(colors))
    actual_mean = mean(channelview(RGB.(SimpleNeRF.image(view))); dims=(2, 3))
    diff = mean(abs, mean_color .- actual_mean)
    @test Float32(diff) < 1e-5
end