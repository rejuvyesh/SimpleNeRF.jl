module SimpleNeRF

using Random

using Distributions
using Flux
using Configurations

@option struct NeRFConfig
    n_input_layers::Int=5
    n_mid_layers::Int=4
    hiddendim::Int=256
    color_layer_dim::Int=128
    x_freqs::Int=10
    d_freqs::Int=4
end

struct NeRFModel
    input_layers
    mid_layers
    density
    color
    x_freqs::Int
    d_freqs::Int
end

Flux.@functor NeRFModel (input_layers, mid_layers, density, color,)


function sinusodial_emb(coords, freqs::Int)
    coeffs = 2 .^ range(0, freqs)
    inputs = reshape(coords, 1, size(coords)...) * coeffs
    sines = sin.(inputs)
    cosines = cos.(inputs)
    combined = vcat(sines, cosines)
    return reshape(combined, :, size(combined)[begin+2:end]...)
end

function NeRFModel(;config::NeRFConfig)
    input_layers = Chain()
    mid_layers = Chain()

    density = Chain(Dense(hiddendim, 1), softplus)
    color = Chain(Dense(hiddendim+ddim, config.color_layer_dim), relu, Dense(config.color_layer_dim, 3), tanh_fast)
    return NeRFModel(input_layers, mid_layers, density, color, config.x_freqs, config.d_freqs)
end

function (m::NeRFModel)(x, d)
    x_emb = sinusodial_emb(x, m.x_freqs)
    d_emb = sinusodial_emb(d, m.d_freqs)

    z = m.input_layers(x_emb)
    z1 = vcat(z, x_emb)
    z2 = m.mid_layers(z1)
    density = m.density(z2)
    z3 = vcat(z2, d_emb)
    rgb = m.color(z3)
    return density, rgb
end


struct NeRFRenderer
    coarse::NeRFModel
    fine::NeRFModel
end

function render_rays(rng::AbstractRNG, nr::NeRFRenderer, batch)
end

struct RaySamples
    t_min
    t_max
    mask
    ts
end

function stratified_sampling(t_min, t_mask, count::Int; rng=Random.GLOBAL_RNG)
    bin_size = (t_max .- t_min) ./ count
    bin_size = reshape(bin_size, 1, size(bin_size)...)
    tmp = collect(range(0, count))
    bin_starts = reshape(tmp, 1, size(tmp)...) * bin_size .+ reshape(t_min, 1, size(t_min)...)
    randoms = rand(rng, Uniform(), size(bin_starts)) .* bin_size
    ts = bin_starts + randoms
    return RaySamples(t_min, t_max, mask, ts)
end

function points(rs::RaySamples, rays)
    return rays[begin:begin, :] + (rays[begin+1:end, :] * reshape(rs.ts, 1, size(rs.ts)...))
end

function starts(rs::RaySamples)
    t_mid = (rs.ts[begin+1:end,:] .+ rs.ts[begin:end-1,:]) ./ 2
    return vcat(reshape(rs.t_min, 1, size(rs.t_min)...), t_mid)
end
function ends(rs::RaySamples)
end

function deltas(rs::RaySamples)
    ends(rs) .- starts(rs)
end

function termination_probs(rs::RaySamples, densities)

end

"""

Volumetric rendering given density and color samples along a batch of rays.
"""
function render_rays(rs::RaySamples, densities::AbstractMatrix{T}, rgbs::AbstractArray{T,3}, background::AbstractVector{T}) where {T<:AbstractFloat}
end

function fine_sampling(rng::AbstractRNG, rs::RaySamples, count::Int, densities; combine::Bool=true, eps=Float32(1e-8))
    w = termination_probs(rs, densities)[begin:end-1, :] .+ eps

    # Setup an inverse CDF for inverse transform sampling


    # Evaluate the inverse CDF at quasi-random points.


    return RaySamples(rs.t_min, rs.t_max, rs.mask, new_ts)
end



end
