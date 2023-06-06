using HiddenMarkovModels
using HiddenMarkovModels: MyDiagNormal
using HiddenMarkovModels: rand_prob_vec, rand_trans_mat, sum_to_one!
using LogarithmicNumbers
using Test: @test

N = 10
D = 100  # observations

# True model

p = rand_prob_vec(N);
A = rand_trans_mat(N);
transitions = StandardTransitions(p, A)
emissions = VectorEmissions([MyDiagNormal(float(i), 1.0, D) for i in 1:N])
hmm = HMM(transitions, emissions)

# Simulation

(; state_seq, obs_seq) = rand(hmm, 100);

# Learning

## Without logarithmic

p_init = rand_prob_vec(N);
A_init = rand_trans_mat(N);
transitions_init = StandardTransitions(p_init, A_init)
emissions_init = VectorEmissions([MyDiagNormal(rand(), 1.0, D) for i in 1:N])
hmm_init = HMM(transitions_init, emissions_init)

@test_throws OverflowError baum_welch(hmm_init, [obs_seq]; max_iterations=100, rtol=NaN);

## With logarithmic

p_init_log = rand_prob_vec(N);
A_init_log = rand_trans_mat(N);
transitions_init_log = StandardTransitions(p_init, A_init)
emissions_init_log = VectorEmissions([
    MyDiagNormal(rand(), ULogarithmic(1.0), D) for i in 1:N
])
hmm_init_log = HMM(transitions_init_log, emissions_init_log)

hmm_est_log, logL_evolution = @inferred baum_welch(
    hmm_init_log, [obs_seq]; max_iterations=100, rtol=NaN
);
@test typeof(hmm_est_log) == typeof(hmm_init_log)
