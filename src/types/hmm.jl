"""
$(TYPEDEF)

Basic implementation of an HMM.

# Fields

$(TYPEDFIELDS)
"""
struct HiddenMarkovModel{D,U<:AbstractVector,M<:AbstractMatrix,V<:AbstractVector{D}} <:
       AbstractHMM
    "initial state probabilities"
    init::U
    "state transition matrix"
    trans::M
    "observation distributions"
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

function Base.eltype(hmm::HMM, obs)
    init_type = eltype(hmm.init)
    trans_type = eltype(hmm.trans)
    logdensity_type = typeof(logdensityof(hmm.dists[1], obs))
    return promote_type(init_type, trans_type, logdensity_type)
end

initialization(hmm::HMM) = hmm.init
transition_matrix(hmm::HMM) = hmm.trans
obs_distributions(hmm::HMM) = hmm.dists

function StatsAPI.fit!(hmm::HMM, bw::BaumWelchStorage)
    @unpack fbs, state_marginals_concat, obs_seqs_concat = bw
    # Initialization
    hmm.init .= zero(eltype(hmm.init))
    for k in eachindex(fbs)
        hmm.init .+= view(fbs[k].γ, :, 1)
    end
    sum_to_one!(hmm.init)
    # Transition matrix
    hmm.trans .= zero(eltype(hmm.trans))
    for k in eachindex(fbs)
        sum!(hmm.trans, fbs[k].ξ; init=false)
    end
    foreach(sum_to_one!, eachrow(hmm.trans))
    #  Observation distributions
    for i in eachindex(hmm.dists)
        fit_element_from_sequence!(
            hmm.dists, i, obs_seqs_concat, view(state_marginals_concat, i, :)
        )
    end
    check_hmm(hmm)
    return nothing
end
