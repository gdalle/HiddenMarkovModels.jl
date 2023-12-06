# # Periodic HMM

using Distributions
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using SimpleUnPack
using StatsAPI
using Test  #src

# ## Structure

struct PeriodicHMM{L,T<:Number,D} <: AbstractHMM
    init::Vector{T}
    trans_periodic::NTuple{L,Matrix{T}}
    dists_periodic::NTuple{L,Vector{D}}
end

#-

function HMMs.initialization(hmm::PeriodicHMM)
    return hmm.init
end

function HMMs.transition_matrix(hmm::PeriodicHMM{L}, t::Integer) where {L}
    return hmm.trans_periodic[(t - 1) % L + 1]
end

function HMMs.obs_distributions(hmm::PeriodicHMM{L}, t::Integer) where {L}
    return hmm.dists_periodic[(t - 1) % L + 1]
end

# Construction

init = [0.4, 0.6]
trans_periodic = ([0.8 0.2; 0.2 0.8], [0.6 0.4; 0.4 0.6])
dists_periodic = ([Normal(-1), Normal(+1)], [Normal(-2), Normal(+2)])
hmm = PeriodicHMM(init, trans_periodic, dists_periodic)

# Simulation

control_seq = 1:100
state_seq, obs_seq = rand(hmm, control_seq)

# Inference

viterbi(hmm, obs_seq, control_seq)
forward(hmm, obs_seq, control_seq)
logdensityof(hmm, obs_seq, control_seq)
forward_backward(hmm, obs_seq, control_seq)

## Fitting

function fit_states!(hmm::PeriodicHMM{L}, bw_storage::HMMs.BaumWelchStorage)
    @unpack fb_storages, obs_seqs_concat, state_marginals_concat, seq_limits = bw_storage
    K = length(fb_storages)
    N = length(hmm)

    hmm.init .= zero(eltype(hmm.init))
    for l in 1:L
        hmm.trans_periodic[l] .= zero(eltype(hmm.trans_periodic[l]))
    end
    for k in 1:K
        @unpack γ, ξ = fb_storages[k]
        T = size(γ, 2)
        hmm.init .+= view(γ, :, 1)
        for t in 1:(T - 1)
            hmm.trans_periodic[(t - 1) % L + 1] .+= ξ[t]
        end
    end
    hmm.init ./= sum(hmm.init)
    for l in 1:L, i in 1:N
        @views hmm.trans_periodic[l][i, :] ./= sum(hmm.trans_periodic[l][i, :])
    end
end

function fit_observations!(hmm::PeriodicHMM{L}, bw_storage::HMMs.BaumWelchStorage)
    @unpack fb_storages, obs_seqs_concat, state_marginals_concat, seq_limits = bw_storage
    K = length(fb_storages)
    N = length(hmm)

    for l in 1:L
        indices_l = reduce(vcat, (seq_limits[k] + l):L:seq_limits[k + 1] for k in 1:K)
        obs_seq_periodic = view(obs_seqs_concat, indices_l)
        state_marginals_periodic = view(state_marginals_concat, :, indices_l)
        for i in 1:N
            HMMs.fit_element_from_sequence!(
                hmm.dists_periodic[l],
                i,
                obs_seq_periodic,
                view(state_marginals_periodic, i, :),
            )
        end
    end
end

function StatsAPI.fit!(
    hmm::PeriodicHMM{L}, bw_storage::HMMs.BaumWelchStorage, args...
) where {L}
    fit_states!(hmm, bw_storage)
    fit_observations!(hmm, bw_storage)
    return nothing
end

# Test

init_guess = [0.5, 0.5]
trans_periodic_guess = ([0.7 0.3; 0.3 0.7], [0.5 0.5; 0.5 0.5])
dists_periodic_guess = ([Normal(-0.7), Normal(+0.7)], [Normal(-1.5), Normal(+1.5)])
hmm_guess = PeriodicHMM(init_guess, trans_periodic_guess, dists_periodic_guess)

control_seqs = [1:rand(1000:2000) for _ in 1:10]
obs_seqs = [rand(hmm, control_seq).obs_seq for control_seq in control_seqs]

hmm_est, logL_evolution = baum_welch(hmm_guess, MultiSeq(obs_seqs), MultiSeq(control_seqs))

@test HMMs.similar_hmms(hmm_est, hmm, 1:2; atol=0.05)  #src
