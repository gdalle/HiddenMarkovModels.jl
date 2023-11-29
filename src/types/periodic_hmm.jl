"""
$(TYPEDEF)

Basic implementation of a time-heterogeneous HMM with periodic transition matrices and observation distributions.

# Fields

$(TYPEDFIELDS)
"""
struct PeriodicHiddenMarkovModel{
    period<:Integer,V<:AbstractVector,M<:AbstractMatrix,VD<:AbstractVector
} <: AbstractHMM
    "initial state probabilities"
    init::V
    "one state transition matrix per time"
    trans_periodic::NTuple{period,M}
    "one vector of observation distributions per time (must be amenable to `logdensityof` and `rand`)"
    dists_periodic::NTuple{period,VD}
end

"""
    PeriodicHMM

Alias for the type `HiddenMarkovModel`.
"""
const PeriodicHMM = PeriodicHiddenMarkovModel

function Base.copy(phmm::PeriodicHMM)
    return PeriodicHMM(
        copy(phmm.init), copy(phmm.trans_periodic), copy(phmm.dists_periodic)
    )
end

Base.length(phmm::PeriodicHMM) = length(phmm.init)
initialization(phmm::PeriodicHMM) = phmm.init

function transition_matrix(phmm::PeriodicHMM{period}, t::Integer) where {period}
    return phmm.trans_periodic[(t - 1) % period + 1]
end

function obs_distributions(phmm::PeriodicHMM{period}, t::Integer) where {period}
    return phmm.dists_periodic[(t - 1) % period + 1]
end

function StatsAPI.fit!(
    phmm::PeriodicHMM,
    init_count::Vector,
    trans_count::AbstractMatrix,
    obs_seq::Vector,
    state_marginals::Matrix,
)
    return nothing
end

function permute(phmm::PeriodicHMM{period}, perm::Vector{Int}) where {period}
    return PeriodicHMM(
        phmm.init[perm],
        SVector{period}(trans[perm, :][:, perm] for trans in phmm.trans_periodic),
        SVector{period}(dists[perm] for dists in phmm.dists_periodic),
    )
end
