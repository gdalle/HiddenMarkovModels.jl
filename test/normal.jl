using HiddenMarkovModels
using HiddenMarkovModels: MyNormal, rand_prob_vec, rand_trans_mat, likelihoods
using Random
using Statistics
using Test

N = 3
p = rand_prob_vec(N)
A = rand_trans_mat(N)
em = [MyNormal(float(i), 1.0) for i in 1:N]

hmm = HMM(p, A, em)
θ = nothing

(state_seq, obs_seq) = rand(hmm, θ, 100)

likelihoods(hmm, θ, obs_seq)
forward_backward(hmm, θ, obs_seq)

p_init = rand_prob_vec(N)
A_init = rand_trans_mat(N)
em_init = [MyNormal(float(i+1), 2.0) for i in 1:N]
hmm_init = HMM(p_init, A_init, em_init)

obs_seqs = [last(rand(hmm, θ, 1000)) for k in 1:100];

hmm_est, logL_evolution = baum_welch(hmm_init, obs_seqs)
