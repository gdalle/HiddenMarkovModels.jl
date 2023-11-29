"""
$(TYPEDEF)

Basic implementation of a time-heterogeneous HMM with periodic transition matrices and observation distributions. 

The period is the first type parameter `L`.

# Fields

$(TYPEDFIELDS)
"""
struct PeriodicHMM{L,V<:AbstractVector,M<:AbstractMatrix,VD<:AbstractVector} <: AbstractHMM
    "initial state probabilities"
    init::V
    "one state transition matrix per time"
    trans_periodic::NTuple{L,M}
    "one vector of observation distributions per time (must be amenable to `logdensityof` and `rand`)"
    dists_periodic::NTuple{L,VD}
end

function Base.copy(phmm::PeriodicHMM)
    return PeriodicHMM(
        copy(phmm.init), copy(phmm.trans_periodic), copy(phmm.dists_periodic)
    )
end

Base.length(phmm::PeriodicHMM) = length(phmm.init)
initialization(phmm::PeriodicHMM) = phmm.init

function transition_matrix(phmm::PeriodicHMM{L}, t::Integer) where {L}
    return phmm.trans_periodic[(t - 1) % L + 1]
end

function obs_distributions(phmm::PeriodicHMM{L}, t::Integer) where {L}
    return phmm.dists_periodic[(t - 1) % L + 1]
end

## Fitting

struct BaumWelchStoragePeriodicHMM{L,O,R} <: AbstractBaumWelchStorage
    obs_seqs_concat_periodic::NTuple{L,Vector{O}}
    state_marginals_concat_periodic::NTuple{L,Matrix{R}}
end

function initialize_baum_welch(
    ::PeriodicHMM{L},
    fb_storages::Vector{<:ForwardBackwardStorage},
    obs_seqs::Vector{<:Vector},
) where {L}
    obs_seqs_concat_periodic = ntuple(
        l -> reduce(vcat, obs_seqs[k][l:L:end] for k in eachindex(obs_seqs)), L
    )
    state_marginals_concat_periodic = ntuple(
        l -> reduce(hcat, fb_storages[k].γ[:, l:L:end] for k in eachindex(fb_storages)), L
    )
    return BaumWelchStoragePeriodicHMM(
        obs_seqs_concat_periodic, state_marginals_concat_periodic
    )
end

function update_baum_welch!(
    bw_storage::BaumWelchStoragePeriodicHMM{L},
    fb_storages::Vector{<:ForwardBackwardStorage},
    obs_seqs::Vector{<:Vector},
) where {L}
    @unpack state_marginals_concat_periodic = bw_storage
    for l in 1:L
        tl = 1
        for k in eachindex(obs_seqs, fb_storages)
            @unpack γ = fb_storages[k]
            γl = @view γ[:, l:L:end]
            Tl = size(γl, 2)
            state_marginals_concat_periodic[l][:, tl:(tl + Tl - 1)] .= γl
            tl += Tl
        end
    end
    return nothing
end

function fit_states!(
    hmm::PeriodicHMM{L}, fb_storages::Vector{<:ForwardBackwardStorage}
) where {L}
    # Reset
    hmm.init .= zero(eltype(hmm.init))
    for l in 1:L
        hmm.trans_periodic[l] .= zero(eltype(hmm.trans_periodic[l]))
    end
    # Accumulate sufficient stats
    for k in eachindex(fb_storages)
        @unpack γ, ξ = fb_storages[k]
        hmm.init .+= view(γ, :, 1)
        for t in eachindex(ξ)
            l = (t - 1) % L + 1
            mynonzeros(hmm.trans_periodic[l]) .+= mynonzeros(ξ[t])
        end
    end
    # Normalize
    sum_to_one!(hmm.init)
    for l in 1:L
        foreach(sum_to_one!, eachrow(hmm.trans_periodic[l]))
    end
    return nothing
end

function fit_observations!(
    hmm::PeriodicHMM{L}, bw_storage::BaumWelchStoragePeriodicHMM
) where {L}
    @unpack obs_seqs_concat_periodic, state_marginals_concat_periodic = bw_storage
    # Fit observation distributions
    for l in 1:L
        for i in 1:length(hmm)
            fit_element_from_sequence!(
                hmm.dists_periodic[l],
                i,
                obs_seqs_concat_periodic[l],
                view(state_marginals_concat_periodic[l], i, :),
            )
        end
    end
    return nothing
end

function StatsAPI.fit!(
    hmm::PeriodicHMM{L},
    bw_storage::BaumWelchStoragePeriodicHMM,
    fb_storages::Vector{<:ForwardBackwardStorage},
    obs_seqs::Vector{<:Vector},
) where {L}
    update_baum_welch!(bw_storage, fb_storages, obs_seqs)
    fit_states!(hmm, fb_storages)
    fit_observations!(hmm, bw_storage)
    return nothing
end
