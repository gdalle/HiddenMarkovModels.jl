module HMMComparison

using BenchmarkTools: BenchmarkGroup, @benchmarkable
using CondaPkg: CondaPkg
using Distributions: Normal, MvNormal
using HiddenMarkovModels: HiddenMarkovModels
using HMMBase: HMMBase
using LinearAlgebra: Diagonal
using LogExpFunctions: logsumexp
using PythonCall: Py, PyArray, pyimport, pyconvert, pybuiltins, pylist
using Random: AbstractRNG
using Reexport: @reexport
using SparseArrays: spdiagm
@reexport using HMMBenchmark

export HMMBaseImplem, hmmlearnImplem, pomegranateImplem, dynamaxImplem
export print_python_setup, compare_loglikelihoods

include("hmmbase.jl")
include("hmmlearn.jl")
include("pomegranate.jl")
include("dynamax.jl")
include("setup.jl")
include("correctness.jl")

end # module HMMComparison
