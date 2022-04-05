# From https://gist.github.com/mcabbott/80ac43cca3bee8f57809155a5240519f
function _repeat(x::AbstractArray, counts::Integer...)
    N = max(ndims(x), length(counts))
    size_y = ntuple(d -> size(x,d) * get(counts, d, 1), N)
    size_x2 = ntuple(d -> isodd(d) ? size(x, 1+d÷2) : 1, 2*N)
  
    ## version without mutation
    # ignores = ntuple(d -> reshape(Base.OneTo(counts[d]), ntuple(_->1, 2d-1)..., :), length(counts))
    # y = reshape(broadcast(first∘tuple, reshape(x, size_x2), ignores...), size_y)
  
    # ## version with mutation
    size_y2 = ntuple(d -> isodd(d) ? size(x, 1+d÷2) : get(counts, d÷2, 1), 2*N)
    y = similar(x, size_y)
    reshape(y, size_y2) .= reshape(x, size_x2)
    y
end
  
function ChainRulesCore.rrule(::typeof(_repeat), x::AbstractArray, counts::Integer...)
    size_x = size(x)
    function repeat_pullback_1(dy_raw)
        dy = unthunk(dy_raw)
        size2ndims = ntuple(d -> isodd(d) ? get(size_x, 1+d÷2, 1) : get(counts, d÷2, 1), 2*ndims(dy))
        reduced = sum(reshape(dy, size2ndims); dims = ntuple(d -> 2d, ndims(dy)))
        return (NoTangent(), reshape(reduced, size_x), map(_->NoTangent(), counts)...)
    end
    return _repeat(x, counts...), repeat_pullback_1
end
  