"""
    HiddenMarkovModels

A Julia package for HMM modeling, simulation, inference and learning.
"""
module HiddenMarkovModels

const HMMs = HiddenMarkovModels

using ChainRulesCore: ChainRulesCore
using DensityInterface: DensityInterface, DensityKind, HasDensity, NoDensity
using DensityInterface: densityof, logdensityof
using Distributions: Categorical, fit_mle
using LinearAlgebra: Diagonal, dot, mul!
using Random: AbstractRNG, GLOBAL_RNG

include("utils.jl")
include("abstract.jl")
include("hmm.jl")
include("simulation.jl")
include("likelihoods.jl")
include("forward_backward.jl")
include("baum_welch.jl")
include("mynormal.jl")

export HMMs
export AbstractHiddenMarkovModel, AbstractHMM
export HiddenMarkovModel, HMM
export forward_backward
export baum_welch!

end
