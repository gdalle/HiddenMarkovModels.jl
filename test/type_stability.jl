using Distributions: Normal
using HiddenMarkovModels
using HiddenMarkovModels: MyNormal, rand_prob_vec, rand_trans_mat
using JET: @test_opt, @test_call
using Test: @inferred

N = 5

# True model

sp = StandardStateProcess(rand_prob_vec(N), rand_trans_mat(N))
op = StandardObservationProcess([MyNormal(randn(), 1.0) for i in 1:N])
hmm = HMM(sp, op)

# Simulation

(; state_seq, obs_seq) = rand(hmm, 100);

# Inference

@inferred viterbi(hmm, obs_seq);
@test_opt viterbi(hmm, obs_seq)
@test_call viterbi(hmm, obs_seq)

@inferred forward_backward(hmm, obs_seq);
@test_opt target_modules = (HiddenMarkovModels,) forward_backward(hmm, obs_seq)
@test_call forward_backward(hmm, obs_seq)

# Learning

hmm_init = copy(hmm)

@inferred baum_welch(hmm_init, [obs_seq]; rtol=NaN);
@test_opt target_modules = (HiddenMarkovModels,) baum_welch(hmm_init, [obs_seq]; rtol=NaN)
@test_call baum_welch(hmm_init, [obs_seq]; rtol=NaN)
