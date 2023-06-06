using Distributions: Normal
using HiddenMarkovModels
using HiddenMarkovModels: rand_prob_vec, rand_trans_mat, sum_to_one!
using JET: @test_opt, @test_call
using Test: @inferred

N = 10

# True model

p = rand_prob_vec(N);
A = rand_trans_mat(N);
transitions = StandardTransitions(p, A)
emissions = VectorEmissions([Normal(float(i), 1.0) for i in 1:N])
hmm = HMM(transitions, emissions)

# Simulation

(; state_seq, obs_seq) = rand(hmm, 100);
obs_seqs = [rand(hmm, 20).obs_seq for k in 1:10];

# Inference

@inferred viterbi(hmm, obs_seq);
@test_opt viterbi(hmm, obs_seq)
@test_call viterbi(hmm, obs_seq)

@inferred forward_backward(hmm, obs_seq);
@test_opt target_modules = (HiddenMarkovModels,) forward_backward(hmm, obs_seq)
@test_call forward_backward(hmm, obs_seq)

# Learning

p_init = rand_prob_vec(N);
A_init = rand_trans_mat(N);
transitions_init = StandardTransitions(p_init, A_init)
emissions_init = VectorEmissions([Normal(rand(), 1.0) for i in 1:N])
hmm_init = HMM(transitions_init, emissions_init)

@inferred baum_welch(hmm_init, obs_seqs; rtol=0);
@test_opt target_modules = (HiddenMarkovModels,) baum_welch(hmm_init, obs_seqs; rtol=0)
@test_call baum_welch(hmm_init, obs_seqs; rtol=0)
