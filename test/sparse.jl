using Distributions: Normal
using HiddenMarkovModels
using HiddenMarkovModels: sum_to_one!
using LinearAlgebra
using SparseArrays
using Test

N = 5

p = ones(N) / N
A = SparseMatrixCSC(SymTridiagonal(ones(N), ones(N - 1)))
foreach(sum_to_one!, eachrow(A))

sp = StandardStateProcess(p, A)
op = StandardObservationProcess([Normal(randn(), 1.0) for i in 1:N])
hmm = HMM(sp, op)

(; state_seq, obs_seq) = rand(hmm, 100)
hmm_est, logL_evolution = @inferred baum_welch(hmm, [obs_seq])

@test typeof(hmm_est) == typeof(hmm)
@test nnz(transition_matrix(hmm_est.state_process)) <=
    nnz(transition_matrix(hmm.state_process))
