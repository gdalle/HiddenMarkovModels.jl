using HiddenMarkovModels
using HiddenMarkovModels: LightDiagNormal
using LinearAlgebra
using LogarithmicNumbers
using SimpleUnPack
using Test

N, D, T = 3, 2, 1000

## LogarithmicNumbers

p = ones(N) / N;
A = rand_trans_mat(N);
d = [LightDiagNormal(randn(D), ones(D)) for i in 1:N];
d_init = [LightDiagNormal(randn(D), ones(D)) for i in 1:N];
d_init_log = [LightDiagNormal(randn(D), LogFloat64.(ones(D))) for i in 1:N];

hmm = HMM(p, A, d);
obs_seq = rand(hmm, T).obs_seq;

hmm_init1 = HMM(LogFloat64.(p), A, d_init);
hmm_est1, logL_evolution1 = @inferred baum_welch(hmm_init1, obs_seq);

hmm_init2 = HMM(p, LogFloat64.(A), d_init);
hmm_est2, logL_evolution2 = @inferred baum_welch(hmm_init2, obs_seq);

hmm_init3 = HMM(p, A, d_init_log);
hmm_est3, logL_evolution3 = @inferred baum_welch(hmm_init3, obs_seq);

@testset "Logarithmic" begin
    @test typeof(hmm_est1) == typeof(hmm_init1)
    @test typeof(hmm_est2) == typeof(hmm_init2)
    @test typeof(hmm_est3) == typeof(hmm_init3)
end
