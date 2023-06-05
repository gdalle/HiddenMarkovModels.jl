using Aqua
using Documenter
using HiddenMarkovModels
using JuliaFormatter
using Test

DocMeta.setdocmeta!(
    HiddenMarkovModels, :DocTestSetup, :(using HiddenMarkovModels); recursive=true
)

@testset verbose = true "HiddenMarkovModels.jl" begin
    @testset verbose = false "Code quality" begin
        Aqua.test_all(HiddenMarkovModels; ambiguities=false)
    end

    @testset verbose = true "Code formatting" begin
        @test format(HiddenMarkovModels; verbose=false, overwrite=false)
    end

    doctest(HiddenMarkovModels)

    @testset verbose = true "Correctness" begin
        include("correctness.jl")
    end

    @testset verbose = true "Efficiency" begin
        include("efficiency.jl")
    end
end
