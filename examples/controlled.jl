# # Controlled HMM

using Distributions
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using LinearAlgebra
using Random
using SimpleUnPack
using StatsAPI
using Test  #src

#-

rng = Random.default_rng()
Random.seed!(rng, 63)

#-

struct ControlledGaussianHMM{T<:Number} <: AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dist_coeffs::Vector{Vector{T}}
end

#-

function HMMs.initialization(hmm::ControlledGaussianHMM)
    return hmm.init
end

function HMMs.transition_matrix(hmm::ControlledGaussianHMM, control::AbstractVector)
    return hmm.trans
end

function HMMs.obs_distributions(hmm::ControlledGaussianHMM, control::AbstractVector)
    return [Normal(dot(control, hmm.dist_coeffs[i])) for i in 1:length(hmm)]
end

#-

init = [0.4, 0.6]
trans = [0.7 0.3; 0.2 0.8]
dist_coeffs = [-ones(3), ones(3)]
hmm = ControlledGaussianHMM(init, trans, dist_coeffs)

#-

T = 100
control_seq = [randn(rng, 3) for t in 1:T];
state_seq, obs_seq = rand(rng, hmm, control_seq);

#-

viterbi(hmm, obs_seq, control_seq)
forward(hmm, obs_seq, control_seq)
logdensityof(hmm, obs_seq, control_seq)
forward_backward(hmm, obs_seq, control_seq);

#-

function fit_states!(
    hmm::ControlledGaussianHMM{T}, bw_storage::HMMs.BaumWelchStorage
) where {T}
    @unpack fb_storages, obs_seqs_concat, state_marginals_concat, seq_limits = bw_storage
    K, N = length(fb_storages), length(hmm)
    hmm.init .= zero(T)
    hmm.trans .= zero(T)
    for k in 1:K
        @unpack γ, ξ = fb_storages[k]
        @views hmm.init .+= γ[:, 1]
        hmm.trans .+= sum(ξ)
    end
    hmm.init ./= sum(hmm.init)
    for i in 1:N
        @views hmm.trans[i, :] ./= sum(hmm.trans[i, :])
    end
end

function fit_observations!(hmm::ControlledGaussianHMM, bw_storage::HMMs.BaumWelchStorage)
    @unpack fb_storages,
    obs_seqs_concat, control_seqs_concat, state_marginals_concat,
    seq_limits, = bw_storage
    N = length(hmm)
    X = transpose(stack(control_seqs_concat))
    y = stack(obs_seqs_concat)
    for i in 1:N
        W = sqrt.(Diagonal(view(state_marginals_concat, i, :)))
        hmm.dist_coeffs[i] = (W * X) \ (W * y)
    end
end

function StatsAPI.fit!(
    hmm::ControlledGaussianHMM, bw_storage::HMMs.BaumWelchStorage, args...
)
    fit_states!(hmm, bw_storage)
    fit_observations!(hmm, bw_storage)
    return nothing
end

#-

init_guess = [0.5, 0.5]
trans_guess = [0.5 0.5; 0.5 0.5]
dist_coeffs_guess = [-0.5 * ones(3), 0.5 * ones(3)]
hmm_guess = ControlledGaussianHMM(init_guess, trans_guess, dist_coeffs_guess)

#-

control_seqs = [[randn(rng, 3) for t in 1:rand(T:(2T))] for k in 1:100];
obs_seqs = [rand(rng, hmm, control_seq).obs_seq for control_seq in control_seqs];

hmm_est, logL_evolution = baum_welch(hmm_guess, MultiSeq(obs_seqs), MultiSeq(control_seqs))
@test HMMs.similar_hmms(hmm_est, hmm, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; atol=0.1)  #src
