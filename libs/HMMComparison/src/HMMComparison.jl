module HMMComparison

using BenchmarkTools: BenchmarkGroup, @benchmarkable
using CondaPkg: CondaPkg
using Distributions: Normal, MvNormal
using HMMBase: HMMBase
using HMMBenchmark:
    HMMBenchmark,
    Instance,
    Implementation,
    HiddenMarkovModelsImplem,
    build_params,
    build_model,
    build_benchmarkables
using LinearAlgebra: Diagonal
using PythonCall: pyimport, pybuiltins
using Random: AbstractRNG
using SparseArrays: spdiagm

export HiddenMarkovModelsImplem,
    HMMBaseImplem, hmmlearnImplem, pomegranateImplem, dynamaxImplem
export build_model, build_benchmarkables

include("hmmbase.jl")
include("hmmlearn.jl")
include("pomegranate.jl")
include("dynamax.jl")
include("setup.jl")

end # module HMMComparison
