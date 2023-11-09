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

    @testset "Doctests" begin
        Documenter.doctest(HiddenMarkovModels)
    end

    @testset "Correctness" begin
        include("correctness.jl")
    end

    @testset "Array types" begin
        include("arrays.jl")
    end

    @testset "Number types" begin
        include("numbers.jl")
    end

    @testset "Autodiff" begin
        include("autodiff.jl")
    end

    @testset "DNA" begin
        include("dna.jl")
    end

    @testset "Misc" begin
        include("misc.jl")
    end
end
