struct NeRFRenderer
    coarse::NeRFModel
    fine::NeRFModel
end

Flux.@functor NeRFRenderer 

function render_rays(nr::NeRFRenderer, batch; rng::AbstractRNG=Random.GLOBAL_RNG)
    t_min, t_max, mask = batched_t_range(batch)
    coarse_ts = stratified_sampling(t_min, t_max, mask, coarse_ts; rng)
    all_points = points(coarse_ts, batch)
    direction_batch = repeat(batch[:, 2:3, :], (1, size(all_points, 1), 1))
    coarse_densities, coarse_rgbs = nr.coarse(all_points, direction_batch)

    coarse_outputs = render_rays(coarse_ts, coarse_densities, coarse_rgbs, background)

    fine_ts = fine_sampling(coarse_ts, fine_ts, coarse_densities; rng)
    all_points = points(fine_ts, batch)
    fine_densities, fine_rgbs = nr.fine(all_points, direction_bath)

    fine_outputs = render_rays(fine_ts, fine_densities, fine_rgbs, background)
    return (coarse=coarse_outputs, fine=fine_outputs)
end

function batched_t_range(batch; bbox_min, bbox_max, eps=1e-8)
    # batch: 3x2xB
    origin = batch[:, 1, :]
    direction = batch[:, 2, :]

    offsets = bbox .- origin # 2 x 3 x B
    ts = offsets ./ (direction .+ eps)
    ts = hcat(minimum(ts, dims=2), maximum(ts, dims=2))

    # Find overlapping and apply constraints
    max_of_min = maximum(ts, dims=2)
    min_t = relu(max_of_min)
    max_t = minimum(ts, dims=2)
    max_t_clipped = max.(max_t, min_t .+ min_t_range)
    real_range = stack([min_t, max_t_clipped], dims=1)
    null_range = repeat([0, min_t_range], size(real_range)[end])
    mask = min_t .< max_t
    bounds = ifelse.(mask, real_range, null_range)
    return bounds[:, 1, :], bounds[:, 2, :], mask
end

struct RaySamples
    t_min
    t_max
    mask
    ts
end

function stratified_sampling(t_min::AbstractVector, t_max::AbstractVector, mask, count::Int; rng=Random.GLOBAL_RNG)
    bin_size = (t_max .- t_min) ./ count
    bin_size = reshape(bin_size, 1, size(bin_size)...)
    tmp = collect(range(0, count))
    bin_starts = reshape(tmp, 1, size(tmp)...) .* bin_size .+ reshape(t_min, 1, size(t_min)...)
    @check size(bin_starts) == (count, size(t_min, 1))
    randoms = rand(rng, Uniform(), size(bin_starts)) .* bin_size
    ts = bin_starts .+ randoms
    return RaySamples(t_min, t_max, mask, ts)
end

"""
For each ray, compute the points at all timesteps

- `rays`: 3x2xB batch of rays

Returns a batch of points of shape 3xTxB
"""
function points(rs::RaySamples, rays)
    return rays[begin:begin, :] + (rays[begin+1:end, :] .* reshape(rs.ts, 1, size(rs.ts)...))
end

function starts(rs::RaySamples)
    t_mid = (rs.ts[begin+1:end,:] .+ rs.ts[begin:end-1,:]) ./ 2
    return vcat(reshape(rs.t_min, 1, size(rs.t_min)...), t_mid)
end

function ends(rs::RaySamples)
    t_mid = (rs.ts[begin+1:end, :] .+ rs.ts[begin:end-1, :]) ./ 2
    return vcat(t_mid, reshape(rs.t_max, 1, size(rs.t_max)...))
end

function deltas(rs::RaySamples)
    ends(rs) .- starts(rs)
end

function termination_probs(rs::RaySamples, densities)
    density_dt = densities .* deltas(rs)

    # Integral of termination probabilities over time.
    acc_densities_cur = cumsum(density_dt, dims=1) # TODO check
    acc_densities_prev = vcat(zeros(eltype(densities), 1, size(acc_densities_cur,2)), acc_densities_cur)
    prob_survive = exp.(-acc_densities_prev)

    # Probability of terminating at time t given we made it to time t
    prob_terminate = vcat((1 .- exp.(-density_dt)), ones(eltype(densities), 1, size(rs.ts, 1)))

    return prob_survive .* prob_terminate
end

"""

Volumetric rendering given density and color samples along a batch of rays.
"""
function render_rays(rs::RaySamples, densities::AbstractMatrix{T}, rgbs::AbstractArray{T,3}, background::AbstractVector{T}) where {T<:AbstractFloat}
    probs = termination_probs(rs, densities)
    colors = hcat(rgbs, repeat(reshape(background, size(background, 1), 1, 1) , 1, 1, size(rgbs)[end])) # TODO
    return @. ifelse(reshape(rs.mask, 1, size(rs.mask)...), sum(reshape(probs, 1, size(probs)...) .* colors; dims=1), background)
end

function fine_sampling(rs::RaySamples, count::Int, densities; combine::Bool=true, eps=Float32(1e-8), rng::AbstractRNG=Random.GLOBAL_RNG)
    w = termination_probs(rs, densities)[begin:end-1, :] .+ eps

    # Setup an inverse CDF for inverse transform sampling
    xs = cumsum(w; dims=1)
    xs = vcat(zeros(eltype(densities), 1, size(rs.ts, 1)), xs)
    xs = xs ./ xs[end:end, :]
    ys = vcat(reshape(rs.t_min, 1, size(rs.t_min)...), ends(rs))

    # Evaluate the inverse CDF at quasi-random points.
    input_samples = stratified_sampling(t_min, t_max, rs.mask, count; rng=rng)
    new_ts = batched_interpolate(input_samples.ts, xs, ys)

    if combine
        combined = vcat(rs.ts, new_ts)
        new_ts = sort(combined; dims=1)
    end
    return RaySamples(rs.t_min, rs.t_max, rs.mask, new_ts)
end


