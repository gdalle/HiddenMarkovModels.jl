using Aqua: Aqua
using Documenter: Documenter
using HiddenMarkovModels
using JuliaFormatter: JuliaFormatter
using JET: JET
using Test

@testset verbose = true "HiddenMarkovModels.jl" begin
    @testset "Code formatting" begin
        @test JuliaFormatter.format(HiddenMarkovModels; verbose=false, overwrite=false)
    end

    if VERSION >= v"1.9"
        @testset "Code quality" begin
            Aqua.test_all(HiddenMarkovModels; ambiguities=false)
        end

        @testset "Code linting" begin
            JET.test_package(HiddenMarkovModels; target_defined_modules=true)
        end

        @testset verbose = true "Type stability" begin
            include("type_stability.jl")
        end

        @testset verbose = true "Allocations" begin
            include("allocations.jl")
        end
    end

    @testset "Interface" begin
        nothing
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

    @testset verbose = true "Logarithmic" begin
        include("logarithmic.jl")
    end

    @testset verbose = true "Autodiff" begin
        include("autodiff.jl")
    end

    @testset verbose = true "DNA" begin
        include("dna.jl")
    end

    @testset verbose = true "Permuted" begin
        include("permuted.jl")
    end

    @testset "Doctests" begin
        Documenter.doctest(HiddenMarkovModels)
    end
end
