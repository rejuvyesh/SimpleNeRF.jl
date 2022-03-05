using SimpleNeRF: sinusodial_emb

bs = 16
freq = 4
coords = randn(Flaot32, 3, bs)
comb = sinusoidal_emb(coords, freq)
@test size(comb) = (2*3*freq, bs)