using PyCallChainRules.Jax: JaxFunctionWrapper, jax

# TODO: figure out our own
function batched_interpolate(ts, xs, ys)
    JaxFunctionWrapper(jax.jit(jax.vmap(jax.numpy.interp)))(ts, xs, ys)
end