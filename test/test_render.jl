using Test
using SimpleNeRF
bbox_max = [1, 1, 1]
bbox_min = [-1, -1, -1]
background = -1*ones(Float64, 3)
bs = 4
batch = ones(Float32, 3, 2, bs)
t_min, t_max, mask = SimpleNeRF.batched_t_range(batch; bbox_min, bbox_max)
@test size(t_min) == size(t_max) == size(mask) == (bs,)

coarse_ts_c = 64
coarse_ts = SimpleNeRF.stratified_sampling(t_min, t_max, mask, coarse_ts_c)
@test size(coarse_ts.ts) == (coarse_ts_c, bs)
all_points = SimpleNeRF.points(coarse_ts, batch)
dbatch = repeat(batch[:, 2:2, :], 1, size(all_points, 2), 1)

coarse_model = SimpleNeRF.NeRFModel(;config=SimpleNeRF.NeRFConfig())
coarse_densities, coarse_rgbs = coarse_model(all_points, dbatch)
coarse_densities = dropdims(coarse_densities, dims=1)

coarse_outs = SimpleNeRF.render_rays(coarse_ts, coarse_densities, coarse_rgbs, background)

fine_ts_c = 128
fine_ts = SimpleNeRF.fine_sampling(coarse_ts, fine_ts_c, coarse_densities)
all_points = SimpleNeRF.points(fine_ts, batch)
dbatch = repeat(batch[:, 2:2, :], 1, size(all_points, 2), 1)

fine_model = SimpleNeRF.NeRFModel(;config=SimpleNeRF.NeRFConfig())
fine_densities, fine_rgbs = fine_model(all_points, dbatch)
fine_densities = dropdims(fine_densities, dims=1)
fine_outs = SimpleNeRF.render_rays(fine_ts, fine_densities, fine_rgbs, background)
