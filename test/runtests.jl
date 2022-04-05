using SimpleNeRF
using Test

@testset "SimpleNeRF.jl" begin
    # Write your tests here.
    @testset "data" begin
        include("test_data.jl")
    end
    @testset "model" begin
        include("test_model.jl")
    end
    @testset "render" begin
        include("test_render.jl")
    end
end
