# # Built-in HMM

# ## Setup

# Imports

using Distributions
using HiddenMarkovModels
using Random

# Random seed

rng = Random.default_rng()
Random.seed!(rng, 63)

# ## Model creation

N = 2
init = rand_prob_vec(N)
trans = rand_trans_mat(N)
dists = [Normal(i, 0.5) for i in 1:N]
hmm = HMM(init, trans, dists)

# ## Simulation

T = 100
state_seq, obs_seq = rand(rng, hmm, T)

# ## Inference on a single sequence

viterbi(hmm, obs_seq)

#-

logdensityof(hmm, obs_seq)

#-

forward(hmm, obs_seq)

#-

forward_backward(hmm, obs_seq)

# ## Learning from several sequences

K = 3
obs_seqs = [rand(rng, hmm, k * T).obs_seq for k in 1:K]

# Baum-Welch needs an initial guess

init_guess = ones(N) / N
trans_guess = ones(N, N) / N
dists_guess = [Normal(i, 1) for i in 1:N]
hmm_guess = HMM(init_guess, trans_guess, dists_guess)

#-

hmm_est, logL_evolution = baum_welch(hmm_guess, obs_seqs, length(obs_seqs))

first(logL_evolution), last(logL_evolution)

cat(hmm_est.trans, hmm.trans; dims=3)
