using Aqua
using Documenter
using HiddenMarkovModels
using JuliaFormatter
using Test

DocMeta.setdocmeta!(
    HiddenMarkovModels, :DocTestSetup, :(using HiddenMarkovModels); recursive=true
)

@testset verbose = true "HiddenMarkovModels.jl" begin
    @testset verbose = true "Code quality" begin
        Aqua.test_all(HiddenMarkovModels; ambiguities=false)
    end

    @testset verbose = true "Code formatting" begin
        @test format(HiddenMarkovModels; verbose=true, overwrite=false)
    end

    doctest(HiddenMarkovModels)

    @testset verbose = true "Normal HMM" begin
        include("normal.jl")
    end
end
