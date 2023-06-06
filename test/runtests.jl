using Test

@testset verbose = true "HiddenMarkovModels.jl" begin
    @testset "Code quality" begin
        include("quality.jl")
    end

    @testset "Code formatting" begin
        include("formatting.jl")
    end

    @testset verbose = true "Type stability" begin
        include("type_stability.jl")
    end

    @testset verbose = true "Correctness" begin
        include("correctness.jl")
    end

    @testset verbose = true "Sparse" begin
        include("sparse.jl")
    end

    @testset verbose = true "Logarithmic" begin
        include("logarithmic.jl")
    end
end
