# ## Structure

struct PeriodicHMM{V,VM,VVD} <: AbstractHMM
    init::V
    trans_periodic::VM
    dists_periodic::VVD
end

#-

period(hmm::PeriodicHMM) = length(hmm.trans_periodic)

Base.length(phmm::PeriodicHMM) = length(phmm.init)
HMMs.initialization(phmm::PeriodicHMM) = phmm.init

function HMMs.transition_matrix(phmm::PeriodicHMM, t::Integer)
    return phmm.trans_periodic[(t - 1) % period(hmm) + 1]
end

function HMMs.obs_distributions(phmm::PeriodicHMM, t::Integer)
    return phmm.dists_periodic[(t - 1) % period(hmm) + 1]
end

## Fitting

function fit_states!(hmm::PeriodicHMM, fb_storages::Vector{<:HMMs.ForwardBackwardStorage})
    L = period(hmm)
    hmm.init .= 0
    for l in 1:L
        hmm.trans_periodic[l] .= 0
    end
    for k in eachindex(fb_storages)
        @unpack γ, ξ = fb_storages[k]
        hmm.init .+= view(γ, :, 1)
        for t in eachindex(ξ)
            l = (t - 1) % L + 1
            hmm.trans_periodic[l] .+= ξ[t]
        end
    end
    hmm.init ./= sum(hmm.init)
    for l in 1:L
        hmm.trans_periodic[l] ./= sum(hmm.trans_periodic[l]; dims=2)
    end
    return nothing
end

#-

function fit_observations!(
    hmm::PeriodicHMM,
    fb_storages::Vector{<:HMMs.ForwardBackwardStorage},
    obs_seqs::Vector{<:Vector},
)
    L = period(hmm)
    for l in 1:L
        for i in 1:length(hmm)
            obs_seq_periodic = reduce(
                vcat, obs_seqs[k][l:L:end] for k in eachindex(obs_seqs)
            )
            state_marginals_periodic = reduce(
                vcat, fb_storages[k].γ[i, l:L:end] for k in eachindex(fb_storages)
            )
            D = typeof(hmm.dists_periodic[l][i])
            hmm.dists_periodic[l][i] = fit(D, obs_seq_periodic, state_marginals_periodic)
        end
    end
    return nothing
end

#-

function StatsAPI.fit!(
    hmm::PeriodicHMM,
    ::BaumWelchStoragePeriodicHMM,
    fb_storages::Vector{<:HMMs.ForwardBackwardStorage},
    obs_seqs::Vector{<:Vector},
)
    fit_states!(hmm, fb_storages)
    fit_observations!(hmm, fb_storages, obs_seqs)
    return nothing
end

# ## Example

N = 2
T = 1000

init = ones(N) / N;
trans_periodic = (
    [0.9 0.1; 0.1 0.9], #
    [0.8 0.2; 0.2 0.8], #
    [0.7 0.3; 0.3 0.7],
);
dists_periodic = (
    [Normal(0), Normal(4)], #
    [Normal(2), Normal(6)], #
    [Normal(4), Normal(8)],
);

hmm = PeriodicHMM(init, trans_periodic, dists_periodic);

#-

state_seq, obs_seq = rand(hmm, T);
hmm_est, logL_evolution = baum_welch(hmm, obs_seq);

#md plot(logL_evolution)

#-

cat(hmm_est.init, hmm.init; dims=3)

#-

cat(hmm_est.trans_periodic[1], hmm.trans_periodic[1]; dims=3)
cat(hmm_est.trans_periodic[2], hmm.trans_periodic[2]; dims=3)
cat(hmm_est.trans_periodic[3], hmm.trans_periodic[3]; dims=3)

#-

cat(hmm_est.dists_periodic[1], hmm.dists_periodic[1]; dims=3)
cat(hmm_est.dists_periodic[2], hmm.dists_periodic[2]; dims=3)
cat(hmm_est.dists_periodic[3], hmm.dists_periodic[3]; dims=3)
