using Distributions
using HiddenMarkovModels
using Random

rng = Random.default_rng()
Random.seed!(rng, 63)

N = 3
init = ones(N) / N
trans = rand_trans_mat(N)
dists = [Normal(i, 1) for i in 1:N]

hmm = HMM(init, trans, dists)

T = 100
state_seq, obs_seq = rand(rng, hmm, T)

viterbi(hmm, obs_seq)
logdensityof(hmm, obs_seq)
forward(hmm, obs_seq)
forward_backward(hmm, obs_seq)

baum_welch(hmm, obs_seq)
