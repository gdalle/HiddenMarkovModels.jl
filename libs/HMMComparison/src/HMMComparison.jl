module HMMComparison

using BenchmarkTools
using CondaPkg
using Distributions
using HMMBase
using HMMBenchmark
using HMMBenchmark:
    Implementation,
    HiddenMarkovModelsImplem,
    build_params,
    build_model,
    build_benchmarkables
using LinearAlgebra: Diagonal
using PythonCall
using Random: AbstractRNG
using SparseArrays: spdiagm

export HiddenMarkovModelsImplem,
    HMMBaseImplem, hmmlearnImplem, pomegranateImplem, dynamaxImplem
export build_model, build_benchmarkables
export define_full_suite

include("hmmbase.jl")
include("hmmlearn.jl")
include("pomegranate.jl")
include("dynamax.jl")
include("setup.jl")

end # module HMMComparison
