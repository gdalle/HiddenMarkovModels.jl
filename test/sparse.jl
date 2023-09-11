using Distributions
using HiddenMarkovModels
using HiddenMarkovModels: sum_to_one!
using LinearAlgebra
using SparseArrays
using SimpleUnPack
using Test

N = 5
T = 100

p = ones(N) / N
A = SparseMatrixCSC(SymTridiagonal(ones(N), ones(N - 1)))
foreach(sum_to_one!, eachrow(A))
dists = [Normal(randn(), 1.0) for i in 1:N]
dists_init = [Normal(randn(), 1.0) for i in 1:N]

hmm = HMM(p, A, dists)
hmm_init = HMM(p, A, dists_init)

@unpack state_seq, obs_seq = rand(hmm, T)
hmm_est, logL_evolution = @inferred baum_welch(hmm_init, obs_seq)

@test typeof(hmm_est) == typeof(hmm)
@test nnz(transition_matrix(hmm_est)) <= nnz(transition_matrix(hmm))
