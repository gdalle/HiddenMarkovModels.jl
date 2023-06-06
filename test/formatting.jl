using HiddenMarkovModels
using JuliaFormatter: format
using Test: @test

@test format(HiddenMarkovModels; verbose=false, overwrite=false)
