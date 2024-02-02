module HMMBenchmark

using Base.Threads
using BenchmarkTools: @benchmarkable, BenchmarkGroup
using CSV: CSV
using DataFrames: DataFrame
using Distributions: Normal, MvNormal
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
    baum_welch!,
    duration
using LinearAlgebra: BLAS, Diagonal, SymTridiagonal
using Pkg: Pkg
using Random: AbstractRNG
using SparseArrays: spdiagm
using Statistics: mean, median, std

export AbstractImplementation, Instance
export define_suite, parse_results, print_julia_setup

abstract type Implementation end
Base.string(implem::Implementation) = string(typeof(implem))[begin:(end - length("Implem"))]

include("instance.jl")
include("params.jl")
include("hiddenmarkovmodels.jl")
include("suite.jl")

end
