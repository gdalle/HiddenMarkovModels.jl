using HiddenMarkovModels
using Test

N = 5
T = 100

p = rand_prob_vec(N)
p_rand = rand_prob_vec(N)

A = rand_trans_mat(N)
A_rand = rand_trans_mat(N)

mc = MC(p, A)
mc_rand = MC(p_rand, A_rand)

state_seq = rand(mc, T)

mc_est = fit(mc_rand, state_seq)

@test logdensityof(mc_est, state_seq) >
    logdensityof(mc, state_seq) >
    logdensityof(mc_rand, state_seq)
