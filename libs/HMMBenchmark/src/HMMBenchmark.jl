module HMMBenchmark

using BenchmarkTools: @benchmarkable, BenchmarkGroup
using CSV: CSV
using DataFrames: DataFrame
using Distributions: Normal, DiagNormal, PDiagMat
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
using LinearAlgebra: SymTridiagonal
using Pkg: Pkg
using SimpleUnPack: @unpack

export Configuration, define_suite, parse_results

include("configuration.jl")
include("algos.jl")
include("suite.jl")

benchmarkables_hmmbase(args...; kwargs...) = error("HMMBase not loaded")
benchmarkables_hmmlearn(args...; kwargs...) = error("PythonCall not loaded")
benchmarkables_pomegranate(args...; kwargs...) = error("PythonCall not loaded")

end
