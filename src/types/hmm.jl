"""
$(TYPEDEF)

Basic implementation of an HMM.

# Fields

$(TYPEDFIELDS)
"""
struct HiddenMarkovModel{I<:AbstractVector,T<:AbstractMatrix,D<:AbstractVector} <:
       AbstractHMM
    "initial state probabilities"
    init::I
    "state transition matrix"
    trans::T
    "observation distributions"
    dists::D

    function HiddenMarkovModel(init::I, trans::T, dists::D) where {I,T,D}
        hmm = new{I,T,D}(init, trans, dists)
        check_hmm(hmm)
        return hmm
    end
end

"""
    HMM

Alias for the type `HiddenMarkovModel`.
"""
const HMM = HiddenMarkovModel

function Base.copy(hmm::HMM)
    return HiddenMarkovModel(copy(hmm.init), copy(hmm.trans), copy(hmm.dists))
end

Base.length(hmm::HMM) = length(hmm.init)
initialization(hmm::HMM) = hmm.init
transition_matrix(hmm::HMM) = hmm.trans
obs_distributions(hmm::HMM) = hmm.dists

function StatsAPI.fit!(
    hmm::HMM,
    init_count::Vector,
    trans_count::AbstractMatrix,
    obs_seq::Vector,
    state_marginals::Matrix,
)
    # Initialization
    hmm.init .= init_count
    sum_to_one!(hmm.init)
    # Transition matrix
    hmm.trans .= trans_count
    foreach(sum_to_one!, eachrow(hmm.trans))
    #  Observation distributions
    for i in eachindex(hmm.dists)
        fit_element_from_sequence!(hmm.dists, i, obs_seq, view(state_marginals, i, :))
    end
    check_hmm(hmm)
    return nothing
end
