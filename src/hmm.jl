"""
    HiddenMarkovModel{D} <: AbstractHMM

Basic implementation of an HMM.

# Fields

- `init::AbstractVector`: initial state probabilities
- `trans::AbstractMatrix`: state transition matrix
- `dists::AbstractVector{D}`: observation distributions

# Constructors

```
HMM(init, trans, dists)
```
"""
struct HiddenMarkovModel{D,U<:AbstractVector,M<:AbstractMatrix,V<:AbstractVector{D}} <:
       AbstractHMM
    init::U
    trans::M
    dists::V

    function HiddenMarkovModel(
        init::U, trans::M, dists::V
    ) where {D,U<:AbstractVector,M<:AbstractMatrix,V<:AbstractVector{D}}
        hmm = new{D,U,M,V}(init, trans, dists)
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
initial_distribution(hmm::HMM) = hmm.init
transition_matrix(hmm::HMM) = hmm.trans
obs_distribution(hmm::HMM, i::Integer) = hmm.dists[i]

function StatsAPI.fit!(hmm::HMM, init_count, trans_count, obs_seq, state_marginals)
    hmm.init .= init_count
    sum_to_one!(hmm.init)
    hmm.trans .= trans_count
    foreach(sum_to_one!, eachrow(hmm.trans))
    @views for i in eachindex(hmm.dists)
        fit_element_from_sequence!(hmm.dists, i, obs_seq, state_marginals[i, :])
    end
    return nothing
end
