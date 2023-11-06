"""
    HiddenMarkovModels

A Julia package for HMM modeling, simulation, inference and learning.

# Exports

$(EXPORTS)
"""
module HiddenMarkovModels

using Base: RefValue
using Base.Threads: @threads
using DensityInterface:
    DensityInterface, DensityKind, HasDensity, NoDensity, densityof, logdensityof
using Distributions:
    Distributions,
    Categorical,
    Distribution,
    UnivariateDistribution,
    MultivariateDistribution,
    MatrixDistribution
using DocStringExtensions
using LinearAlgebra: Diagonal, dot, mul!
using PrecompileTools: @compile_workload, @setup_workload
using Random: Random, AbstractRNG, default_rng
using Requires: @require
using SimpleUnPack: @unpack
using StatsAPI: StatsAPI, fit, fit!

export AbstractHiddenMarkovModel, AbstractHMM
export HiddenMarkovModel, HMM
export rand_prob_vec, rand_trans_mat
export initialization, transition_matrix, obs_distributions
export logdensityof, viterbi, forward, forward_backward, baum_welch
export fit!
export check_hmm

include("types/abstract_hmm.jl")
include("types/permuted_hmm.jl")

include("utils/check.jl")
include("utils/probvec.jl")
include("utils/transmat.jl")
include("utils/fit.jl")
include("utils/lightdiagnormal.jl")

include("inference/forward.jl")
include("inference/viterbi.jl")
include("inference/forward_backward.jl")
include("inference/baum_welch.jl")

include("types/hmm.jl")

if !isdefined(Base, :get_extension)
    function __init__()
        @require HMMBase = "b2b3ca75-8444-5ffa-85e6-af70e2b64fe7" include(
            "../ext/HiddenMarkovModelsHMMBaseExt.jl"
        )
        @require ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4" include(
            "../ext/HiddenMarkovModelsChainRulesCoreExt.jl"
        )
    end
end

# @compile_workload begin
#     N, D, T = 5, 3, 100
#     p = rand_prob_vec(N)
#     A = rand_trans_mat(N)
#     dists = [LightDiagNormal(randn(D), ones(D)) for i in 1:N]
#     hmm = HMM(p, A, dists)

#     obs_seq = rand(hmm, T).obs_seq
#     obs_seqs = [rand(hmm, T).obs_seq for _ in 1:3]
#     nb_seqs = 3

#     logdensityof(hmm, obs_seq)
#     logdensityof(hmm, obs_seqs, nb_seqs)

#     forward(hmm, obs_seq)
#     forward(hmm, obs_seqs, nb_seqs)

#     viterbi(hmm, obs_seq)
#     viterbi(hmm, obs_seqs, nb_seqs)

#     forward_backward(hmm, obs_seq)
#     forward_backward(hmm, obs_seqs, nb_seqs)

#     baum_welch(hmm, obs_seq; max_iterations=2, atol=-Inf)
#     baum_welch(hmm, obs_seqs, nb_seqs; max_iterations=2, atol=-Inf)
# end

end
