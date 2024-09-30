using Aqua: Aqua
using Documenter: Documenter
using HiddenMarkovModels
using JET
using JuliaFormatter: JuliaFormatter
using Pkg
using Test

TEST_SUITE = get(ENV, "JULIA_HMM_TEST_SUITE", "Standard")
if TEST_SUITE == "HMMBase"
    Pkg.add("HMMBase")
    using HMMBase: HMMBase
end

Pkg.develop(; path=joinpath(dirname(@__DIR__), "libs", "HMMTest"))

@testset verbose = true "HiddenMarkovModels.jl" begin
    if TEST_SUITE == "Standard"
        @testset "Code formatting" begin
            @test JuliaFormatter.format(HiddenMarkovModels; verbose=false, overwrite=false)
        end

        @testset "Code quality" begin
            Aqua.test_all(
                HiddenMarkovModels; ambiguities=false, deps_compat=(check_extras=false,)
            )
        end

        @testset "Code linting" begin
            using Distributions
            using Zygote
            if VERSION >= v"1.10"
                JET.test_package(HiddenMarkovModels; target_defined_modules=true)
            end
        end

        @testset "Distributions" begin
            include("distributions.jl")
        end

        examples_path = joinpath(dirname(@__DIR__), "examples")
        for file in readdir(examples_path)
            @testset "Example - $file" begin
                include(joinpath(examples_path, file))
            end
        end

        @testset "Doctests" begin
            Documenter.doctest(HiddenMarkovModels)
        end
    end

    @testset verbose = true "Correctness - $TEST_SUITE" begin
        include("correctness.jl")
    end
end
