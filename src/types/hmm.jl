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
    "observation distributions (must be amenable to `logdensityof` and `rand`)"
    dists::VD
end

function Base.copy(hmm::HMM)
    return HMM(copy(hmm.init), copy(hmm.trans), copy(hmm.dists))
end

Base.length(hmm::HMM) = length(hmm.init)
initialization(hmm::HMM) = hmm.init
transition_matrix(hmm::HMM, ::Integer) = hmm.trans
obs_distributions(hmm::HMM, ::Integer) = hmm.dists

## Fitting

function StatsAPI.fit!(hmm::HMM, bw_storage::BaumWelchStorage, obs_seqs::MultiSeq)
    @unpack fb_storages, obs_seqs_concat, state_marginals_concat = bw_storage
    # Fit states
    hmm.init .= zero(eltype(hmm.init))
    hmm.trans .= zero(eltype(hmm.trans))
    for k in eachindex(fb_storages)
        @unpack γ, ξ = fb_storages[k]
        hmm.init .+= view(γ, :, 1)
        for t in eachindex(ξ)
            mynonzeros(hmm.trans) .+= mynonzeros(ξ[t])
        end
    end
    sum_to_one!(hmm.init)
    foreach(sum_to_one!, eachrow(hmm.trans))
    # Fit observations
    for i in 1:length(hmm)
        fit_element_from_sequence!(
            hmm.dists, i, obs_seqs_concat, view(state_marginals_concat, i, :)
        )
    end
    # Safety check
    check_hmm(hmm)
    return nothing
end
