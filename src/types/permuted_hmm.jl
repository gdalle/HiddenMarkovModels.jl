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

Base.length(permuted::PermutedHMM) = length(permuted.hmm)

initialization(permuted::PermutedHMM) = initialization(permuted.hmm)[permuted.perm]

function transition_matrix(permuted::PermutedHMM, control)
    return transition_matrix(permuted.hmm, control)[permuted.perm, :][:, permuted.perm]
end

function obs_distributions(permuted::PermutedHMM, control)
    return obs_distributions(permuted.hmm, control)[permuted.perm]
end
