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
transitions = StandardTransitions(p, A)
emissions = VectorEmissions([Normal(float(i), 1.0) for i in 1:N])
hmm = HMM(transitions, emissions)

# Simulation

(; state_seq, obs_seq) = rand(hmm, 100);

# Learning

p_init = copy(p);
A_init = copy(A);
transitions_init = StandardTransitions(p_init, A_init)
emissions_init = VectorEmissions([Normal(float(i + 1), 1.0) for i in 1:N])
hmm_init = HMM(transitions_init, emissions_init)

hmm_est, logL_evolution = @inferred baum_welch(
    hmm_init, [obs_seq]; max_iterations=100, rtol=NaN
)
@test nnz(initial_distribution(hmm_est)) <= nnz(initial_distribution(hmm_init))
@test nnz(transition_matrix(hmm_est)) <= nnz(transition_matrix(hmm_init))
