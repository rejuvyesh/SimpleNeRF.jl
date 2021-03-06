using SimpleNeRF: sinusodial_emb, NeRFConfig, NeRFModel, total
using Optimisers
using LinearAlgebra
using Test

bs = 16
freq = 4
coords = randn(Float32, 3, bs)
comb = sinusodial_emb(coords, freq)
@test size(comb) == (2*3*freq, bs)

cfg = NeRFConfig()
model = NeRFModel(;config=cfg)
x = randn(Float32, 3, bs)
d = randn(Float32, 3, bs)
density, rgb = model(x, d)
@test size(density) == (1, bs)
@test size(rgb) == (3, bs)

params = Optimisers.setup(Optimisers.ADAM(), model)
@test sqrt(total(x->sum(x.^2), model)) > 1.0f0