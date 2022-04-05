
# TODO: figure out our own
# function batched_interpolate_py(ts, xs, ys)
#     JaxFunctionWrapper(jax.jit(jax.vmap(jax.numpy.interp)))(ts, xs, ys)
# end

# XXX: this is terrible for performance?
function batched_searchsortedlast(xp, x)
    xpb = unbatch(xp)
    xb = unbatch(x)
    #res = searchsortedlast.(Ref.(xpb), xb)
    res = map(xpb, xb) do up, u
        searchsortedlast.(Ref(up), u)
    end
    stack(res, dims=ndims(x))
end

# TODO: check if this is even correct
function batched_interpolate(x, xp::AbstractMatrix, fp::AbstractMatrix)
    @check size(xp) == size(fp)
    idx = Flux.Zygote.@ignore max.(1, min.(batched_searchsortedlast(xp, x), size(xp, 1) - 1))
    next_idx = idx .+ 1
    df = fp[next_idx] .- fp[idx]
    dx = xp[next_idx] .- xp[idx]
    delta = x .- xp[idx]
    otherfp = fp[idx] .+ (delta ./ dx) .* df
    ffp = fp[idx]
    f = ifelse.(dx .== 0, ffp, otherfp)    
    return f
end