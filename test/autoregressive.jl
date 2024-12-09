using Distributions
using HiddenMarkovModels
const HMMs = HiddenMarkovModels

struct AutoRegressiveGaussianHMM{T} <: AbstractHMM{true}
    init::Vector{T}
    trans::Matrix{T}
    a::Vector{T}
    b::Vector{T}
end

const ARGHMM = AutoRegressiveGaussianHMM

HMMs.initialization(hmm::ARGHMM) = hmm.init
HMMs.transition_matrix(hmm::ARGHMM) = hmm.trans

function HMMs.obs_distributions(hmm::ARGHMM, _control, prev_obs)
    (; a, b) = hmm
    return [Normal(a[i] * prev_obs + b[i], 1.0) for i in 1:length(hmm)]
end
