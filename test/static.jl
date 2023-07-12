using Distributions
using HiddenMarkovModels
using LinearAlgebra
using StaticArrays
using SimpleUnPack
using Test

N = 5
T = 1000

p = MVector{N}(ones(N) / N)
A = MMatrix{N,N}(rand_trans_mat(N))
dists = MVector{N}([Normal(randn(), 1.0) for i in 1:N])
dists_init = MVector{N}([Normal(randn(), 1.0) for i in 1:N])

hmm = HMM(p, A, dists)
hmm_init = HMM(p, A, dists_init)

@unpack state_seq, obs_seq = rand(hmm, T)
hmm_est, logL_evolution = @inferred baum_welch(hmm_init, obs_seq)

@test typeof(hmm_est) == typeof(hmm)
