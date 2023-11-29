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

struct BaumWelchStorageHMM{O,R} <: AbstractBaumWelchStorage
    obs_seqs_concat::Vector{O}
    state_marginals_concat::Matrix{R}
end

function initialize_baum_welch(
    ::HMM, fb_storages::Vector{<:ForwardBackwardStorage}, obs_seqs::Vector{<:Vector}
)
    obs_seqs_concat = reduce(vcat, obs_seqs)
    state_marginals_concat = reduce(hcat, fb_storages[k].γ for k in eachindex(fb_storages))
    return BaumWelchStorageHMM(obs_seqs_concat, state_marginals_concat)
end

function update_baum_welch!(
    bw_storage::BaumWelchStorageHMM,
    fb_storages::Vector{<:ForwardBackwardStorage},
    obs_seqs::Vector{<:Vector},
)
    @unpack state_marginals_concat = bw_storage
    t = 1
    for k in eachindex(obs_seqs, fb_storages)
        @unpack γ = fb_storages[k]
        T = size(γ, 2)
        state_marginals_concat[:, t:(t + T - 1)] .= γ
        t += T
    end
    return nothing
end

function fit_states!(hmm::HMM, fb_storages::Vector{<:ForwardBackwardStorage})
    # Reset
    hmm.init .= zero(eltype(hmm.init))
    hmm.trans .= zero(eltype(hmm.trans))
    # Accumulate sufficient stats
    for k in eachindex(fb_storages)
        @unpack γ, ξ = fb_storages[k]
        hmm.init .+= view(γ, :, 1)
        for t in eachindex(ξ)
            mynonzeros(hmm.trans) .+= mynonzeros(ξ[t])
        end
    end
    # Normalize
    sum_to_one!(hmm.init)
    foreach(sum_to_one!, eachrow(hmm.trans))
    return nothing
end

function fit_observations!(hmm::HMM, bw_storage::BaumWelchStorageHMM)
    @unpack obs_seqs_concat, state_marginals_concat = bw_storage
    for i in 1:length(hmm)
        fit_element_from_sequence!(
            hmm.dists, i, obs_seqs_concat, view(state_marginals_concat, i, :)
        )
    end
    return nothing
end

function StatsAPI.fit!(
    hmm::HMM,
    bw_storage::BaumWelchStorageHMM,
    fb_storages::Vector{<:ForwardBackwardStorage},
    obs_seqs::Vector{<:Vector},
)
    update_baum_welch!(bw_storage, fb_storages, obs_seqs)
    fit_states!(hmm, fb_storages)
    fit_observations!(hmm, bw_storage)
    return nothing
end
