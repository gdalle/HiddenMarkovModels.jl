module HMMBenchmark

using BenchmarkTools
using CSV
using DataFrames
using Distributions
using Distributions: PDiagMat
using HMMBase: HMMBase
using LinearAlgebra
using Pkg
using BenchmarkTools
using HiddenMarkovModels
using SimpleUnPack

export define_suite, run_suite, parse_results

include("hmms.jl")
include("hmmbase.jl")
include("suite.jl")

benchmarkables_hmmlearn(; kwargs...) = error("PythonCall not loaded")
benchmarkables_pomegranate(; kwargs...) = error("PythonCall not loaded")

end
