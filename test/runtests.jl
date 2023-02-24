using Aqua
using Documenter
using HiddenMarkovModels
using JuliaFormatter
using Test

DocMeta.setdocmeta!(
    HiddenMarkovModels, :DocTestSetup, :(using HiddenMarkovModels); recursive=true
)

@testset verbose = true "HiddenMarkovModels.jl" begin
    @testset verbose = true "Code quality (Aqua.jl)" begin
        Aqua.test_all(HiddenMarkovModels; ambiguities=false)
    end

    @testset verbose = true "Code formatting (JuliaFormatter.jl)" begin
        @test format(HiddenMarkovModels; verbose=true, overwrite=false)
    end

    @testset verbose = true "Doctests (Documenter.jl)" begin
        doctest(HiddenMarkovModels)
    end
end
