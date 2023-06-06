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

    @testset verbose = true "SparseArrays" begin
        include("sparse.jl")
    end

    @testset verbose = true "LogarithmicNumbers" begin
        include("logarithmic.jl")
    end

    @testset verbose = true "ForwardDiff" begin
        include("forwarddiff.jl")
    end

    @testset verbose = true "Zygote" begin
        include("zygote.jl")
    end
end
