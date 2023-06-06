"""
    HiddenMarkovModels

A Julia package for HMM modeling, simulation, inference and learning.
"""
module HiddenMarkovModels

"""
    HMMs

Alias for the module `HiddenMarkovModels`.
"""
const HMMs = HiddenMarkovModels

using Base.Threads: @threads
using ChainRulesCore: ChainRulesCore
using DensityInterface: DensityInterface, DensityKind, HasDensity, NoDensity
using DensityInterface: densityof, logdensityof
using Distributions: Categorical
using LinearAlgebra: Diagonal, dot, mul!, normalize!
using Random: AbstractRNG, GLOBAL_RNG
using Requires: @require
using StatsAPI: StatsAPI, fit

include("utils/misc.jl")
include("utils/probvec.jl")
include("utils/transmat.jl")
include("utils/mynormal.jl")
include("utils/mydiagnormal.jl")

include("abstract/state_process.jl")
include("abstract/observation_process.jl")

include("hmm.jl")

include("inference/likelihoods.jl")
include("inference/forward_backward.jl")
include("inference/forward_backward_storage.jl")
include("inference/logdensity.jl")
include("inference/viterbi.jl")

include("learning/sufficient_stats.jl")
include("learning/baum_welch.jl")

include("concrete/standard_state_process.jl")
include("concrete/standard_observation_process.jl")

export HMMs
export initial_distribution, transition_matrix
export HiddenMarkovModel, HMM
export logdensityof
export viterbi
export forward_backward
export baum_welch
export StandardStateProcess
export StandardObservationProcess

if !isdefined(Base, :get_extension)
    function __init__()
        @require HMMBase = "b2b3ca75-8444-5ffa-85e6-af70e2b64fe7" include(
            "../ext/HiddenMarkovModelsHMMBaseExt.jl"
        )
    end
end

end
