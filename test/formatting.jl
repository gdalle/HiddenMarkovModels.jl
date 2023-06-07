using HiddenMarkovModels
using JuliaFormatter: format
using Test

@test format(HiddenMarkovModels; verbose=false, overwrite=false)
