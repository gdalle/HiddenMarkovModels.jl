using Distributions: Normal
using HiddenMarkovModels
using HiddenMarkovModels: rand_prob_vec, rand_trans_mat
using JET
using Test

N = 10

# True model

p = rand_prob_vec(N);
A = rand_trans_mat(N);

transitions = MarkovTransitions(p, A);
emissions = VectorEmissions([Normal(float(i), 1.0) for i in 1:N]);

hmm = HMM(transitions, emissions);

# Simulation

(; state_seq, obs_seq) = rand(hmm, 1000);
obs_seqs = [rand(hmm, 20).obs_seq for k in 1:100];

# Inference

@test_opt viterbi(hmm, obs_seq);
@test_call viterbi(hmm, obs_seq);

@test_skip @test_opt forward_backward(hmm, obs_seq)
@test_call forward_backward(hmm, obs_seq)

# Learning

p_est = rand_prob_vec(N)
A_est = rand_trans_mat(N)

transitions_est = MarkovTransitions(p_est, A_est)
emissions_est = VectorEmissions([Normal(rand(), 1.0) for i in 1:N])

hmm_est = HMM(transitions_est, emissions_est)

@test_skip @test_opt baum_welch!(hmm_est, obs_seqs; rtol=0)
@test_call baum_welch!(hmm_est, obs_seqs; rtol=0)
