"""
$(TYPEDEF)

Basic implementation of an HMM.

# Fields

$(TYPEDFIELDS)
"""
struct HiddenMarkovModel{V<:AbstractVector,M<:AbstractMatrix,VD<:AbstractVector} <:
       AbstractHMM
    "initial state probabilities"
    init::V
    "state transition matrix"
    trans::M
    "observation distributions (must be amenable to `logdensityof` and `rand`)"
    dists::VD

    function HiddenMarkovModel(init::V, trans::M, dists::VD) where {V,M,VD}
        hmm = new{V,M,VD}(init, trans, dists)
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
transition_matrix(hmm::HMM, ::Integer) = hmm.trans
obs_distributions(hmm::HMM, ::Integer) = hmm.dists

function StatsAPI.fit!(
    hmm::HMM,
    fb_storages::Vector{<:ForwardBackwardStorage},
    obs_seqs_concat,
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
    return nothing
end
