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

        @testset "Type stability" begin
            include("type_stability.jl")
        end

        @testset "Allocations" begin
            include("allocations.jl")
        end
    end

    @testset "Correctness" begin
        include("correctness.jl")
    end

    @testset "Sparse" begin
        include("sparse.jl")
    end

    @testset "Static" begin
        include("static.jl")
    end

    @testset "Logarithmic" begin
        include("logarithmic.jl")
    end

    @testset "Autodiff" begin
        include("autodiff.jl")
    end

    @testset "DNA" begin
        include("dna.jl")
    end

    @testset "Permuted" begin
        include("permuted.jl")
    end

    @testset "Doctests" begin
        Documenter.doctest(HiddenMarkovModels)
    end
end
