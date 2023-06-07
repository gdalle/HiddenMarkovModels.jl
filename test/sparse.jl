using Distributions: Normal
using HiddenMarkovModels
using HiddenMarkovModels: MyNormal, sum_to_one!
using JET
using SparseArrays: sprand, nnz
using Test

N = 10

p_sparse = sprand(N, 0.5);
sum_to_one!(p_sparse);
A_sparse = sprand(N, N, 0.5);
A_sparse[:, end] .+= 1;
foreach(sum_to_one!, eachrow(A_sparse))
sp_sparse = StandardStateProcess(p_sparse, A_sparse)
op_sparse = StandardObservationProcess([MyNormal(randn(), 1.0) for i in 1:N])
hmm_sparse = HMM(sp_sparse, op_sparse)

(; state_seq, obs_seq) = rand(hmm, 3)
@test_skip baum_welch(hmm, [obs_seq])
