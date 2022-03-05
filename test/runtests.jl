using SimpleNeRF
using Test

@testset "SimpleNeRF.jl" begin
    # Write your tests here.
    @testset "data" begin
        include("test_data.jl")
    end
end
