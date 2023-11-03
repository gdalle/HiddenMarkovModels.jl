"""
$(TYPEDEF)

Wrapper around an `AbstractHMM` that permutes its states.

This is computationally inefficient and mostly useful for evaluation.

# Fields

$(TYPEDFIELDS)
"""
struct PermutedHMM{H<:AbstractHMM} <: AbstractHMM
    "the old HMM"
    hmm::H
    "a permutation such that state `i` in the new HMM corresponds to state `perm[i]` in the old"
    perm::Vector{Int}
end

Base.length(p::PermutedHMM) = length(p.hmm)

initialization(p::PermutedHMM) = initialization(p.hmm)[p.perm]

function transition_matrix(p::PermutedHMM)
    return transition_matrix(p.hmm)[p.perm, :][:, p.perm]
end

function obs_distributions(p::PermutedHMM)
    return obs_distributions(p.hmm)[p.perm]
end
