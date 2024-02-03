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
    baum_welch!
using LinearAlgebra: BLAS, Diagonal, SymTridiagonal
using Pkg: Pkg
using Random: AbstractRNG
using SparseArrays: spdiagm
using Statistics: mean, median, std, quantile

export Implementation, Instance, Params, HiddenMarkovModelsImplem
export build_params, build_data, build_model, build_benchmarkables
export define_suite, parse_results, read_results, print_julia_setup

abstract type Implementation end

include("instance.jl")
include("params.jl")
include("hiddenmarkovmodels.jl")
include("suite.jl")

end
