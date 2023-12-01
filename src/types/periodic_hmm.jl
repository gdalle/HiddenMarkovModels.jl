# ## Structure

struct PeriodicHMM{V,VM,VVD} <: AbstractHMM
    init::V
    trans_periodic::VM
    dists_periodic::VVD
end

#-

period(hmm::HMM) = 1
period(hmm::PeriodicHMM) = length(hmm.trans_periodic)

Base.length(hmm::PeriodicHMM) = length(hmm.init)
initialization(hmm::PeriodicHMM) = hmm.init

function transition_matrix(hmm::PeriodicHMM, t::Integer)
    return hmm.trans_periodic[(t - 1) % period(hmm) + 1]
end

function obs_distributions(hmm::PeriodicHMM, t::Integer)
    return hmm.dists_periodic[(t - 1) % period(hmm) + 1]
end

## Fitting

function StatsAPI.fit!(hmm::PeriodicHMM, bw_storage::BaumWelchStorage, obs_seqs::MultiSeq)
    @unpack fb_storages, obs_seqs_concat, state_marginals_concat, seq_limits = bw_storage
    L = period(hmm)
    # States
    hmm.init .= zero(eltype(hmm.init))
    for l in 1:L
        hmm.trans_periodic[l] .= zero(eltype(hmm.trans_periodic[l]))
    end
    for k in eachindex(fb_storages)
        @unpack γ, ξ = fb_storages[k]
        hmm.init .+= view(γ, :, 1)
        for t in eachindex(ξ)
            l = (t - 1) % L + 1
            mynonzeros(hmm.trans_periodic[l]) .+= mynonzeros(ξ[t])
        end
    end
    sum_to_one!(hmm.init)
    for l in 1:L
        foreach(sum_to_one!, eachrow(hmm.trans_periodic[l]))
    end
    # Observations
    for l in 1:L
        indices_l = reduce(
            vcat, (seq_limits[k] + l):L:seq_limits[k + 1] for k in eachindex(obs_seqs)
        )  # TODO: only allocating line if I'm right
        obs_seq_periodic = view(obs_seqs_concat, indices_l)
        state_marginals_periodic = view(state_marginals_concat, :, indices_l)
        for i in 1:length(hmm)
            fit_element_from_sequence!(
                hmm.dists_periodic[l],
                i,
                obs_seq_periodic,
                view(state_marginals_periodic, i, :),
            )
        end
    end
    return nothing
end
