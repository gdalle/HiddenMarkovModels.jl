using Distributions
using HiddenMarkovModels
using HiddenMarkovModels: LightDiagNormal
using LinearAlgebra
using LogarithmicNumbers
using SimpleUnPack
using Test

N = 3
D = 2
T = 100

p = ones(N) / N
A = rand_trans_mat(N)
dists = [LightDiagNormal(randn(D), ones(D)) for i in 1:N];
dists_init = [LightDiagNormal(randn(D), ones(D)) for i in 1:N];
dists_init_log = [LightDiagNormal(randn(D), LogFloat64.(ones(D))) for i in 1:N];

hmm = HMM(p, A, dists);
@unpack state_seq, obs_seq = rand(hmm, T);

hmm_init = HMM(LogFloat64.(p), A, dists_init);
hmm_est, logL_evolution = @inferred baum_welch(hmm_init, obs_seq);
@test typeof(hmm_est) == typeof(hmm_init)

hmm_init = HMM(p, LogFloat64.(A), dists_init);
hmm_est, logL_evolution = @inferred baum_welch(hmm_init, obs_seq);
@test typeof(hmm_est) == typeof(hmm_init)

hmm_init = HMM(p, A, dists_init_log);
hmm_est, logL_evolution = @inferred baum_welch(hmm_init, obs_seq);
@test typeof(hmm_est) == typeof(hmm_init)
