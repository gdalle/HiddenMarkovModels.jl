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
            Aqua.test_all(HiddenMarkovModels; deps_compat=(check_extras=false,))
        end

        @testset "Code linting" begin
            JET.test_package(HiddenMarkovModels; target_defined_modules=true)
        end

        @testset "Types and allocations" begin
            include("types_allocations.jl")
        end
    end

    @testset "Distributions" begin
        include("distributions.jl")
    end

    @testset "Correctness" begin
        include("correctness.jl")
    end

    @testset "Autodiff" begin
        include("autodiff.jl")
    end

    @testset "Doctests" begin
        Documenter.doctest(HiddenMarkovModels)
    end
end
