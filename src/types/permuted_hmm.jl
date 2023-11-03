"""
    PermutedHMM{H<:AbstractHMM}

Wrapper around an `AbstractHMM` that permutes its states.

This is computationally inefficient and mostly useful for evaluation.

# Fields

- `hmm:H`: the old HMM
- `perm::Vector{Int}`: a permutation such that state `i` in the new HMM corresponds to state `perm[i]` in the old.
"""
struct PermutedHMM{H<:AbstractHMM} <: AbstractHMM
    hmm::H
    perm::Vector{Int}
end

Base.length(p::PermutedHMM) = length(p.hmm)

initial_distribution(p::PermutedHMM) = initial_distribution(p.hmm)[p.perm]

function transition_matrix(p::PermutedHMM)
    return transition_matrix(p.hmm)[p.perm, :][:, p.perm]
end

function obs_distribution(p::PermutedHMM, i::Integer)
    return obs_distribution(p.hmm, p.perm[i])
end
