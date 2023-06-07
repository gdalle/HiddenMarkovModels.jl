using Distributions: Normal
using HiddenMarkovModels
using HiddenMarkovModels: sum_to_one!
using JET
using SparseArrays: sprand, nnz
using Test

N = 10

p = sprand(N, 0.5);
sum_to_one!(p);
A = sprand(N, N, 0.5);
A[:, end] .+= 1;
foreach(sum_to_one!, eachrow(A))
sp = StandardStateProcess(p, A)
op = StandardObservationProcess([Normal(randn(), 1.0) for i in 1:N])
hmm = HMM(sp, op)
hmm_init = copy(hmm)

(; state_seq, obs_seq) = rand(hmm, 100);

@inferred logdensityof(hmm, obs_seq);
@test_opt target_modules = (HMMs,) logdensityof(hmm, obs_seq)
@test_call logdensityof(hmm, obs_seq)

@inferred viterbi(hmm, obs_seq);
@test_opt target_modules = (HMMs,) viterbi(hmm, obs_seq)
@test_call viterbi(hmm, obs_seq)

@inferred forward_backward(hmm, obs_seq);
@test_opt target_modules = (HMMs,) forward_backward(hmm, obs_seq)
@test_call forward_backward(hmm, obs_seq)

@inferred baum_welch(hmm_init, [obs_seq])
@test_opt target_modules = (HMMs,) baum_welch(hmm_init, [obs_seq])
@test_call baum_welch(hmm_init, [obs_seq])

hmm_est, logL_evolution = baum_welch(hmm_init, [obs_seq]; max_iterations=100, rtol=NaN)
@test nnz(initial_distribution(hmm_est.state_process)) <=
    nnz(initial_distribution(hmm_init.state_process))
@test nnz(transition_matrix(hmm_est.state_process)) <=
    nnz(transition_matrix(hmm_init.state_process))
