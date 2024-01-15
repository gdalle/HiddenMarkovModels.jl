"""
    HiddenMarkovModels

A Julia package for HMM modeling, simulation, inference and learning.

# Exports

$(EXPORTS)
"""
module HiddenMarkovModels

using Base: RefValue
using Base.Threads: @threads
using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, rrule_via_ad
using DensityInterface: DensityInterface, DensityKind, HasDensity, NoDensity, logdensityof
using DocStringExtensions
using FillArrays: Fill
using LinearAlgebra: dot, ldiv!, lmul!, mul!
using PrecompileTools: @compile_workload
using Random: Random, AbstractRNG, default_rng
using SparseArrays: AbstractSparseArray, SparseMatrixCSC, nonzeros, nnz, nzrange
using StatsAPI: StatsAPI, fit, fit!

export AbstractHMM, HMM
export initialization, transition_matrix, obs_distributions
export fit!, logdensityof, joint_logdensityof
export viterbi, forward, forward_backward, baum_welch
export seq_limits

include("types/abstract_hmm.jl")

include("utils/linalg.jl")
include("utils/check.jl")
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
