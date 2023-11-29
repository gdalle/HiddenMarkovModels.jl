using Distributions
using HiddenMarkovModels
using Random

rng = Random.default_rng()
Random.seed!(rng, 63)

N = 3
init = ones(N) / N
trans_periodic = (rand_trans_mat(N), rand_trans_mat(N))
dists_periodic = ([Normal(i, 1) for i in 1:N], [Normal(-i, 1) for i in 1:N])

hmm = PeriodicHMM(init, trans_periodic, dists_periodic)

T = 100
state_seq, obs_seq = rand(rng, hmm, T)

viterbi(hmm, obs_seq)
logdensityof(hmm, obs_seq)
forward(hmm, obs_seq)
forward_backward(hmm, obs_seq)

obs_seqs = [rand(rng, hmm, T).obs_seq, rand(rng, hmm, 2T).obs_seq]

baum_welch(hmm, obs_seqs, 2)
