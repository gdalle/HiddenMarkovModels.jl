using Distributions: Normal
using HiddenMarkovModels
using HiddenMarkovModels: rand_prob_vec, rand_trans_mat, sum_to_one!
using SparseArrays: sprand, nnz
using Test: @test

N = 10

# True model

p = sprand(N, 0.8);
sum_to_one!(p);
A = sprand(N, N, 0.8);
foreach(sum_to_one!, eachrow(A))
sp = StandardStateProcess(p, A)
op = StandardObservationProcess([Normal(randn(), 1.0) for i in 1:N])
hmm = HMM(sp, op)

# Simulation

(; state_seq, obs_seq) = rand(hmm, 100);

# Learning

hmm_init = copy(hmm)

hmm_est, logL_evolution = @inferred baum_welch(
    hmm_init, [obs_seq]; max_iterations=100, rtol=NaN
)
@test nnz(initial_distribution(hmm_est.state_process)) <=
    nnz(initial_distribution(hmm_init.state_process))
@test nnz(transition_matrix(hmm_est.state_process)) <=
    nnz(transition_matrix(hmm_init.state_process))
