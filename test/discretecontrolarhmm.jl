using Distributions
using HiddenMarkovModels
const HMMs = HiddenMarkovModels

struct DiscreteCARHMM{T<:Number} <: AbstractHMM{true}
    # Initial distribution P(X_{1}|U_{1}), one vector for each control
    init::Vector{Vector{T}}
    # Transition matrix P(X_{t}|X_{t-1}, U_{t}), one matrix for each control
    trans::Vector{Matrix{T}}
    # Emission matriz P(Y_{t}|X_{t}, U_{t}), one matriz for each control and each possible observation
    dists::Vector{Vector{Matrix{T}}}
    # Prior Distribution for P(Y_{1}|X_{1}, U_{1}), one matriz for each control
    prior::Vector{Matrix{T}}
end

HMMs.initialization(hmm::DiscreteCARHMM, control) = hmm.init[control]

HMMs.transition_matrix(hmm::DiscreteCARHMM, control) = hmm.trans[control]

function HMMs.obs_distributions(hmm::DiscreteCARHMM, control, prev_obs)
    return [Categorical(hmm.dists[control][prev_obs][i, :]) for i in 1:size(hmm, control)]
end

function HMMs.obs_distributions(hmm::DiscreteCARHMM, control, ::Missing)
    return [Categorical(hmm.prior[control][i, :]) for i in 1:size(hmm, control)]
end
