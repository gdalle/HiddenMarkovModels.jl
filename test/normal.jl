using Distributions
using HiddenMarkovModels
using HiddenMarkovModels: rand_prob_vec, rand_trans_mat, loglikelihoods
using Random
using Test

N = 3
p = rand_prob_vec(N)
A = rand_trans_mat(N)
em = [Normal(i) for i in 1:N]

hmm = HMM(p, A, em)

(; state_seq, obs_seq) = rand(hmm, 100)

loglikelihoods(hmm, obs_seq)
