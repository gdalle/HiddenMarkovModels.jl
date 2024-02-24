"""
    HiddenMarkovModels

A Julia package for HMM modeling, simulation, inference and learning.

# Exports

$(EXPORTS)
"""
module HiddenMarkovModels

using ArgCheck: @argcheck
using Base: RefValue
using Base.Threads: @threads
using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, rrule_via_ad
using DensityInterface: DensityInterface, DensityKind, HasDensity, NoDensity, logdensityof
using DocStringExtensions
using FillArrays: Fill
using LinearAlgebra: Transpose, dot, ldiv!, lmul!, mul!, parent
using PrecompileTools: @compile_workload
using Random: Random, AbstractRNG, default_rng
using SparseArrays: AbstractSparseArray, SparseMatrixCSC, nonzeros, nnz, nzrange, rowvals
using StatsAPI: StatsAPI, fit, fit!
using StatsFuns: log2π

export AbstractHMM, HMM
export initialization, transition_matrix, obs_distributions
export fit!, logdensityof, joint_logdensityof
export viterbi, forward, forward_backward, baum_welch
export seq_limits

include("types/abstract_hmm.jl")

include("utils/linalg.jl")
include("utils/valid.jl")
include("utils/probvec_transmat.jl")
include("utils/fit.jl")
include("utils/lightdiagnormal.jl")
include("utils/lightcategorical.jl")
include("utils/limits.jl")

include("inference/forward.jl")
include("inference/viterbi.jl")
include("inference/forward_backward.jl")
include("inference/baum_welch.jl")
include("inference/logdensity.jl")
include("inference/chainrules.jl")

include("types/hmm.jl")

include("precompile.jl")

end
