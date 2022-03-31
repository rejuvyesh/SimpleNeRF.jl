module SimpleNeRF

using Random

using ArgCheck
using Distributions
using Flux
using Functors
using Optimisers
using Configurations
using MLUtils: stack

# From https://github.com/FluxML/Functors.jl/issues/35
# and https://github.com/FluxML/Optimisers.jl/pull/57
const INIT = Base._InitialValue();
function total(f, x; init = INIT)
    fmap(x; exclude = Optimisers.isnumeric, walk = (f, z) -> foreach(f, Optimisers._trainable(z))) do y
      val = f(y)
      init = init===INIT ? val : (init+val)
    end
    init
end

function fmapreduce(f, op, x; init = INIT, walk = (f, x) -> foreach(f, Functors.children(x)), kw...)
    fmap(x; walk, kw...) do y
      init = init===INIT ? f(y) : op(init, f(y))
    end
    init===INIT ? Base.mapreduce_empty(f, op) : init
end

include("data.jl")
include("model.jl")

include("render.jl")

end
