"""
    HiddenMarkovModel{D} <: AbstractHiddenMarkovModel

Basic implementation of an HMM.

# Fields

- `init::AbstractVector`: initial state probabilities
- `trans::AbstractMatrix`: state transition matrix
- `dists::AbstractVector{D}`: observation distributions
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
initialization(hmm::HMM) = hmm.init
transition_matrix(hmm::HMM) = hmm.trans
obs_distributions(hmm::HMM) = hmm.dists

"""
    fit!(hmm::HMM, obs_seqs, fbs)

Update `hmm` in-place using information generated during forward-backward.
"""
function StatsAPI.fit!(hmm::HMM, obs_seqs, fbs)
    # Initial distribution
    hmm.init .= zero(eltype(hmm.init))
    for k in eachindex(fbs)
        @views hmm.init .+= fbs[k].γ[:, 1]
    end
    sum_to_one!(hmm.init)
    # Transition matrix
    hmm.trans .= zero(eltype(hmm.trans))
    for k in eachindex(fbs)
        sum!(hmm.trans, fbs[k].ξ; init=false)
    end
    foreach(sum_to_one!, eachrow(hmm.trans))
    # Observation distributions
    obs_seqs_concat = reduce(vcat, obs_seqs)  # TODO: allocation-free
    state_marginals_concat = reduce(hcat, fb.γ for fb in fbs)  # TODO: allocation-free
    @views for i in eachindex(hmm.dists)
        fit_element_from_sequence!(
            hmm.dists, i, obs_seqs_concat, state_marginals_concat[i, :]
        )
    end
    check_hmm(hmm)
    return nothing
end
