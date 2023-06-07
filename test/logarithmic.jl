using HiddenMarkovModels
using HiddenMarkovModels: MyDiagNormal, rand_prob_vec, rand_trans_mat, sum_to_one!
using LogarithmicNumbers
using Test: @inferred, @test, @test_throws

N = 10
D = 100

# True model

sp = StandardStateProcess(rand_prob_vec(N), rand_trans_mat(N))
op = StandardObservationProcess([MyDiagNormal(randn(D), ones(D)) for i in 1:N])
hmm = HMM(sp, op)

# Simulation

(; state_seq, obs_seq) = rand(hmm, 100);

# Learning

## Without logarithmic

sp_init = StandardStateProcess(rand_prob_vec(N), rand_trans_mat(N))
op_init = StandardObservationProcess([MyDiagNormal(randn(D), ones(D)) for i in 1:N])
hmm_init = HMM(sp_init, op_init)

@test_throws Exception baum_welch(hmm_init, [obs_seq]; max_iterations=100, rtol=NaN);

## With logarithmic

sp_init = StandardStateProcess(rand_prob_vec(N), rand_trans_mat(N))
op_init_log = StandardObservationProcess([
    MyDiagNormal(randn(D), ULogarithmic.(ones(D))) for i in 1:N
])
hmm_init_log = HMM(sp_init, op_init_log)

hmm_est_log, logL_evolution = @inferred baum_welch(
    hmm_init_log, [obs_seq]; max_iterations=100, rtol=NaN
);
@test typeof(hmm_est_log) == typeof(hmm_init_log)
