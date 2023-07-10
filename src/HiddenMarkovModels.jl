"""
    HiddenMarkovModels

A Julia package for HMM modeling, simulation, inference and learning.
"""
module HiddenMarkovModels

"""
    HMMs

Alias for the module HiddenMarkovModels.
"""
const HMMs = HiddenMarkovModels

using Base.Threads: @threads
using ChainRulesCore: ChainRulesCore
using DensityInterface:
    DensityInterface, DensityKind, HasDensity, NoDensity, densityof, logdensityof
using Distributions:
    Distributions,
    Categorical,
    Distribution,
    UnivariateDistribution,
    MultivariateDistribution,
    MatrixDistribution
using LinearAlgebra: Diagonal, dot, mul!
using PrecompileTools: @compile_workload, @setup_workload
using Random: AbstractRNG, GLOBAL_RNG
using Requires: @require
using SimpleUnPack: @unpack
using StatsAPI: StatsAPI, fit, fit!

export HMMs
export AbstractHiddenMarkovModel, AbstractHMM
export HiddenMarkovModel, HMM
export rand_prob_vec, rand_trans_mat
export initial_distribution, transition_matrix, obs_distribution
export logdensityof, viterbi, forward_backward, baum_welch
export LightDiagNormal

include("abstract_hmm.jl")
include("hmm.jl")

include("utils/check.jl")
include("utils/probvec.jl")
include("utils/transmat.jl")
include("utils/fit.jl")
include("utils/lightdiagnormal.jl")

include("inference/loglikelihoods.jl")
include("inference/forward_backward_storage.jl")
include("inference/forward_backward.jl")
include("inference/logdensity.jl")
include("inference/viterbi.jl")

include("learning/sufficient_stats.jl")
include("learning/baum_welch.jl")

if !isdefined(Base, :get_extension)
    function __init__()
        @require HMMBase = "b2b3ca75-8444-5ffa-85e6-af70e2b64fe7" include(
            "../ext/HiddenMarkovModelsHMMBaseExt.jl"
        )
    end
end

@compile_workload begin
    N, D, T = 5, 3, 100
    p = rand_prob_vec(N)
    A = rand_trans_mat(N)
    dists = [LightDiagNormal(randn(D), ones(D)) for i in 1:N]
    hmm = HMM(p, A, dists)

    @unpack state_seq, obs_seq = rand(hmm, T)
    logdensityof(hmm, obs_seq)
    viterbi(hmm, obs_seq)
    forward_backward(hmm, obs_seq)
    baum_welch(hmm, obs_seq; max_iterations=2, rtol=-Inf)
end

end
