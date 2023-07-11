using Test

@testset verbose = true "HiddenMarkovModels.jl" begin
    @testset "Code formatting" begin
        include("formatting.jl")
    end

    if VERSION >= v"1.9"
        @testset "Code quality" begin
            include("quality.jl")
        end

        @testset "Code linting" begin
            include("linting.jl")
        end

        @testset verbose = true "Type stability" begin
            include("type_stability.jl")
        end
    end

    @testset "Interface" begin
        include("interface.jl")
    end

    @testset verbose = true "Correctness" begin
        include("correctness.jl")
    end

    @testset verbose = true "Sparse" begin
        include("sparse.jl")
    end

    @testset verbose = true "Static" begin
        include("static.jl")
    end

    @testset verbose = true "ForwardDiff" begin
        include("forwarddiff.jl")
    end

    @testset "Doctests" begin
        include("doctests.jl")
    end
end
