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
Base.eltype(p::PermutedHMM, obs) = eltype(p.hmm, obs)

initialization(p::PermutedHMM) = initialization(p.hmm)[p.perm]

function transition_matrix(p::PermutedHMM)
    return transition_matrix(p.hmm)[p.perm, :][:, p.perm]
end

function obs_logdensities!(logb::AbstractVector, p::PermutedHMM, obs)
    for i in eachindex(logb, p.dists)
        logb[i] = logdensityof(p.dists[p.perm[i]], obs)
    end
end

function obs_sample(rng::AbstractRNG, p::PermutedHMM, i::Integer)
    return obs_sample(rng, p, p.perm[i])
end
