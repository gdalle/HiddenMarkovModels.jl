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
    "observation distributions (must be amenable to `logdensityof` and `rand`)"
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

function obs_logdensities!(logb::AbstractVector, hmm::HMM, obs)
    for i in eachindex(logb, hmm.dists)
        logb[i] = logdensityof(hmm.dists[i], obs)
    end
end

function obs_sample(rng::AbstractRNG, hmm::HMM, i::Integer)
    return rand(rng, hmm.dists[i])
end

function Base.eltype(hmm::HMM, obs)
    init_type = eltype(hmm.init)
    trans_type = eltype(hmm.trans)
    logdensity_type = typeof(logdensityof(hmm.dists[1], obs))
    return promote_type(init_type, trans_type, logdensity_type)
end

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
