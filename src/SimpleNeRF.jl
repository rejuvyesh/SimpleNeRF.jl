module SimpleNeRF

using Random

using ArgCheck
using Distributions
using Flux
using Configurations

include("data.jl")
include("model.jl")

include("render.jl")

end
