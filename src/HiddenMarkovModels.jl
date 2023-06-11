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
using Random: AbstractRNG, GLOBAL_RNG
using Requires: @require
using StatsAPI: StatsAPI, fit, fit!

export HMMs
export rand_prob_vec, rand_trans_mat
export initial_distribution, transition_matrix, obs_distribution
export HiddenMarkovModel, HMM
export logdensityof, viterbi, forward_backward, baum_welch
export StandardStateProcess, StandardObservationProcess

include("utils/check.jl")
include("utils/probvec.jl")
include("utils/transmat.jl")
include("utils/fit.jl")
include("utils/lightdiagnormal.jl")

include("abstract/state_process.jl")
include("abstract/observation_process.jl")

include("hmm.jl")

include("inference/loglikelihoods.jl")
include("inference/forward_backward_storage.jl")
include("inference/forward_backward.jl")
include("inference/logdensity.jl")
include("inference/viterbi.jl")

include("learning/sufficient_stats.jl")
include("learning/baum_welch.jl")

include("concrete/standard_state_process.jl")
include("concrete/standard_observation_process.jl")

if !isdefined(Base, :get_extension)
    function __init__()
        @require HMMBase = "b2b3ca75-8444-5ffa-85e6-af70e2b64fe7" include(
            "../ext/HiddenMarkovModelsHMMBaseExt.jl"
        )
    end
end

end
