module HMMBenchmark

using BenchmarkTools: @benchmarkable, BenchmarkGroup
using CSV: CSV
using DataFrames: DataFrame
using Distributions: Normal, DiagNormal
using HiddenMarkovModels
using HiddenMarkovModels:
    LightDiagNormal,
    rand_prob_vec,
    rand_trans_mat,
    initialize_viterbi,
    viterbi!,
    initialize_forward,
    forward!,
    initialize_forward_backward,
    forward_backward!,
    baum_welch!
using LinearAlgebra: Diagonal, SymTridiagonal
using Pkg: Pkg
using Random: AbstractRNG

export Configuration, define_suite, parse_results

include("configuration.jl")
include("algos.jl")
include("suite.jl")

benchmarkables_hmmbase(args...; kwargs...) = error("HMMBase not loaded")
benchmarkables_dynamax(args...; kwargs...) = error("PythonCall not loaded")
benchmarkables_hmmlearn(args...; kwargs...) = error("PythonCall not loaded")
benchmarkables_pomegranate(args...; kwargs...) = error("PythonCall not loaded")
print_python_setup(args...; kwargs...) = error("PythonCall not loaded")

end
