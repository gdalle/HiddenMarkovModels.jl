# # Basics

using Distributions
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using Random
using Test  #src

#-

rng = Random.default_rng()
Random.seed!(rng, 63)

#-

init = [0.4, 0.6]
trans = [0.7 0.3; 0.2 0.8]
dists = [Normal(-1, 0.3), Normal(1, 0.3)]
hmm = HMM(init, trans, dists)

#-

T = 100
state_seq, obs_seq = rand(rng, hmm, T)

#-

best_state_seq, _ = viterbi(hmm, obs_seq)

#-

logL = logdensityof(hmm, obs_seq)

#-

last_state_marginals, _ = forward(hmm, obs_seq)

#-

state_marginals, _ = forward_backward(hmm, obs_seq)

#-

init_guess = [0.5, 0.5]
trans_guess = [0.5 0.5; 0.5 0.5]
dists_guess = [Normal(-0.5, 1), Normal(0.5, 1)]
hmm_guess = HMM(init_guess, trans_guess, dists_guess)

#-

obs_seqs = [rand(rng, hmm, rand(T:(2T))).obs_seq for k in 1:100];

hmm_est, logL_evolution = baum_welch(
    hmm_guess, reduce(vcat, obs_seqs); seq_ends=cumsum(length.(obs_seqs))
)

#-

first(logL_evolution), last(logL_evolution)

#-

cat(hmm_est.trans, hmm.trans; dims=3)
@test HMMs.similar_hmms(hmm_est, hmm; atol=1e-1)  #src
