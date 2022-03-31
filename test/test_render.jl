using Test
using SimpleNeRF
bbox_max = [1, 1, 1]
bbox_min = [-1, -1, -1]
bs = 4
batch = ones(Float32, 3, 2, bs)
t_min, t_max, mask = SimpleNeRF.batched_t_range(batch; bbox_min, bbox_max)
@test size(t_min) == size(t_max) == size(mask) == (bs,)