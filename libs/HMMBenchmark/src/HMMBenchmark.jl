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

export Configuration, define_suite, parse_results, print_julia_setup

include("configuration.jl")
include("algos.jl")
include("suite.jl")

end
