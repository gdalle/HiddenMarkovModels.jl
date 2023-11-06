using Distributions
using Distributions: PDiagMat
using HiddenMarkovModels
using HiddenMarkovModels: LightDiagNormal
using JET
using SimpleUnPack
using Test

function test_type_stability(hmm, hmm_init; T)
    obs_seq = rand(hmm, T).obs_seq
    nb_seqs = 2
    obs_seqs = [obs_seq for _ in 1:nb_seqs]

    @testset "Logdensity" begin
        @inferred logdensityof(hmm, obs_seqs, nb_seqs)
        @test_opt target_modules = (HiddenMarkovModels,) logdensityof(
            hmm, obs_seqs, nb_seqs
        )
        @test_call target_modules = (HiddenMarkovModels,) logdensityof(
            hmm, obs_seqs, nb_seqs
        )
    end

    @testset "Forward" begin
        @inferred forward(hmm, obs_seqs, nb_seqs)
        @test_opt target_modules = (HiddenMarkovModels,) forward(hmm, obs_seqs, nb_seqs)
        @test_call target_modules = (HiddenMarkovModels,) forward(hmm, obs_seqs, nb_seqs)
    end

    @testset "Viterbi" begin
        @inferred viterbi(hmm, obs_seqs, nb_seqs)
        @test_opt target_modules = (HiddenMarkovModels,) viterbi(hmm, obs_seqs, nb_seqs)
        @test_call target_modules = (HiddenMarkovModels,) viterbi(hmm, obs_seqs, nb_seqs)
    end

    @testset "Forward-backward" begin
        @inferred forward_backward(hmm, obs_seqs, nb_seqs)
        @test_opt target_modules = (HiddenMarkovModels,) forward_backward(
            hmm, obs_seqs, nb_seqs
        )
        @test_call target_modules = (HiddenMarkovModels,) forward_backward(
            hmm, obs_seqs, nb_seqs
        )
    end

    @testset "Baum-Welch" begin
        @inferred baum_welch(hmm_init, obs_seqs, nb_seqs)
        @test_opt target_modules = (HiddenMarkovModels,) baum_welch(
            hmm_init, obs_seqs, nb_seqs
        )
        @test_call target_modules = (HiddenMarkovModels,) baum_welch(
            hmm_init, obs_seqs, nb_seqs
        )
    end
end

N = 2
D = 3
T = 100

p = rand_prob_vec(N)
p_init = rand_prob_vec(N)

A = rand_trans_mat(N)
A_init = rand_trans_mat(N)

# Normal

dists_norm = [Normal(randn(), 1.0) for i in 1:N]
dists_norm_init = [Normal(randn(), 1) for i in 1:N]

hmm_norm = HMM(p, A, dists_norm)
hmm_norm_init = HMM(p_init, A_init, dists_norm_init)

@testset "Normal" begin
    test_type_stability(hmm_norm, hmm_norm_init; T)
end

# DiagNormal

dists_diagnorm = [DiagNormal(randn(D), PDiagMat(ones(D) .^ 2)) for i in 1:N]
dists_diagnorm_init = [DiagNormal(randn(D), PDiagMat(ones(D) .^ 2)) for i in 1:N]

hmm_diagnorm = HMM(p, A, dists_diagnorm)
hmm_diagnorm_init = HMM(p, A, dists_diagnorm_init)

@testset "DiagNormal" begin
    test_type_stability(hmm_diagnorm, hmm_diagnorm_init; T)
end

## LightDiagNormal

dists_lightdiagnorm = [LightDiagNormal(randn(D), ones(D)) for i in 1:N]
dists_lightdiagnorm_init = [LightDiagNormal(randn(D), ones(D)) for i in 1:N]

hmm_lightdiagnorm = HMM(p, A, dists_lightdiagnorm)
hmm_lightdiagnorm_init = HMM(p, A, dists_lightdiagnorm_init)

@testset "LightDiagNormal" begin
    test_type_stability(hmm_lightdiagnorm, hmm_lightdiagnorm_init; T)
end
