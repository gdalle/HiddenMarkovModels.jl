using HiddenMarkovModels
using HiddenMarkovModels: MyDiagNormal
using HiddenMarkovModels: rand_prob_vec, rand_trans_mat, sum_to_one!
using LogarithmicNumbers
using Test: @inferred, @test, @test_throws

N = 5
D = 1000  # op

# True model

p = rand_prob_vec(N);
A = rand_trans_mat(N);
sp = StandardStateProcess(p, A)
op = StandardObservationProcess([MyDiagNormal(randn(), 1.0, D) for i in 1:N])
hmm = HMM(sp, op)

# Simulation

(; state_seq, obs_seq) = rand(hmm, 100);

# Learning

## Without logarithmic

p_init = rand_prob_vec(N);
A_init = rand_trans_mat(N);
sp_init = StandardStateProcess(p_init, A_init)
op_init = StandardObservationProcess([MyDiagNormal(rand(), 1.0, D) for i in 1:N])
hmm_init = HMM(sp_init, op_init)

@test_throws CompositeException baum_welch(
    hmm_init, [obs_seq]; max_iterations=100, rtol=NaN
);

## With logarithmic

p_init_log = rand_prob_vec(N);
A_init_log = rand_trans_mat(N);
sp_init_log = StandardStateProcess(p_init, A_init)
op_init_log = StandardObservationProcess([
    MyDiagNormal(rand(), ULogarithmic(1.0), D) for i in 1:N
])
hmm_init_log = HMM(sp_init_log, op_init_log)

hmm_est_log, logL_evolution = @inferred baum_welch(
    hmm_init_log, [obs_seq]; max_iterations=100, rtol=NaN
);
@test typeof(hmm_est_log) == typeof(hmm_init_log)
