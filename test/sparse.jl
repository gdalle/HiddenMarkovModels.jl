using Distributions
using HiddenMarkovModels
using HiddenMarkovModels: sum_to_one!
using LinearAlgebra
using SparseArrays
using SimpleUnPack
using Test

N = 3
T = 2000

p = ones(N) / N
A = SparseMatrixCSC(SymTridiagonal(ones(N), ones(N - 1)))
foreach(sum_to_one!, eachrow(A))
dists = [Normal(i + randn(), 1) for i in 1:N]
dists_init = [Normal(i + randn(), 1) for i in 1:N]

hmm = HMM(p, A, dists)
hmm_init = HMM(p, A, dists_init)

obs_seq = rand(hmm, T).obs_seq
hmm_est, logL_evolution = @inferred baum_welch(hmm_init, obs_seq)

@test typeof(hmm_est) == typeof(hmm_init)
@test nnz(transition_matrix(hmm_est)) <= nnz(transition_matrix(hmm))
