"""
    HiddenMarkovModels

A Julia package for HMM modeling, simulation, inference and learning.
"""
module HiddenMarkovModels

using DensityInterface: DensityInterface, densityof, logdensityof
using Distributions: Categorical
using LinearAlgebra
using Random: AbstractRNG, GLOBAL_RNG, rand

include("utils.jl")
include("abstract.jl")
include("hmm.jl")
include("simulation.jl")
include("likelihoods.jl")

export AbstractHiddenMarkovModel, AbstractHMM
export HiddenMarkovModel, HMM

end
