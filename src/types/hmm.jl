"""
$(TYPEDEF)

Basic implementation of an HMM.

# Fields

$(TYPEDFIELDS)
"""
struct HMM{V<:AbstractVector,M<:AbstractMatrix,VD<:AbstractVector} <: AbstractHMM
    "initial state probabilities"
    init::V
    "state transition matrix"
    trans::M
    "observation distributions"
    dists::VD

    function HMM(init::AbstractVector, trans::AbstractMatrix, dists::AbstractVector)
        hmm = new{typeof(init),typeof(trans),typeof(dists)}(init, trans, dists)
        check_hmm(hmm)
        return hmm
    end
end

function Base.copy(hmm::HMM)
    return HMM(copy(hmm.init), copy(hmm.trans), copy(hmm.dists))
end

initialization(hmm::HMM) = hmm.init
transition_matrix(hmm::HMM) = hmm.trans
obs_distributions(hmm::HMM) = hmm.dists

## Fitting

function StatsAPI.fit!(
    hmm::HMM,
    fb_storage::ForwardBackwardStorage,
    obs_seq::AbstractVector;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
)
    (; γ, ξ) = fb_storage
    # Fit states
    hmm.init .= zero(eltype(hmm.init))
    hmm.trans .= zero(eltype(hmm.trans))
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        hmm.init .+= view(γ, :, t1)
        for t in t1:(t2 - 1)
            mynonzeros(hmm.trans) .+= mynonzeros(ξ[t])
        end
    end
    sum_to_one!(hmm.init)
    foreach(sum_to_one!, eachrow(hmm.trans))
    # Fit observations
    for i in 1:length(hmm)
        fit_in_sequence!(hmm.dists, i, obs_seq, view(γ, i, :))
    end
    # Safety check
    check_hmm(hmm)
    return nothing
end
