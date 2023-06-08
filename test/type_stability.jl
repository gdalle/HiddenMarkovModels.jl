using Distributions
using Distributions: PDiagMat
using HiddenMarkovModels
using JET
using Test

function test_type_stability(hmm, hmm_init; T)
    (; state_seq, obs_seq) = rand(hmm, T)

    @testset "Logdensity" begin
        @inferred logdensityof(hmm, obs_seq)
        @test_opt target_modules = (HiddenMarkovModels,) logdensityof(hmm, obs_seq)
        @test_call target_modules = (HiddenMarkovModels,) logdensityof(hmm, obs_seq)
    end

    @testset "Viterbi" begin
        @inferred viterbi(hmm, obs_seq)
        @test_opt target_modules = (HiddenMarkovModels,) viterbi(hmm, obs_seq)
        @test_call target_modules = (HiddenMarkovModels,) viterbi(hmm, obs_seq)
    end

    @testset "Forward-backward" begin
        @inferred forward_backward(hmm, obs_seq)
        @test_opt target_modules = (HiddenMarkovModels,) forward_backward(hmm, obs_seq)
        @test_call target_modules = (HiddenMarkovModels,) forward_backward(hmm, obs_seq)
    end

    @testset "Baum-Welch" begin
        @inferred baum_welch(hmm_init, [obs_seq])
        @test_opt target_modules = (HiddenMarkovModels,) baum_welch(hmm_init, [obs_seq])
        @test_call target_modules = (HiddenMarkovModels,) baum_welch(hmm_init, [obs_seq])
    end
end

N = 5
D = 3

p = rand_prob_vec(N)
p_init = rand_prob_vec(N)

A = rand_trans_mat(N)
A_init = rand_trans_mat(N)

# Normal

dists_norm = [Normal(randn(), 1.0) for i in 1:N]
dists_norm_init = [Normal(randn(), 1) for i in 1:N]

hmm_norm = HMM(p, A, dists_norm)
hmm_norm_init = HMM(p_init, A_init, dists_norm_init)

test_type_stability(hmm_norm, hmm_norm_init; T=100)

# DiagNormal

dists_diagnorm = [DiagNormal(randn(D), PDiagMat(ones(D))) for i in 1:N]
dists_diagnorm_init = [DiagNormal(randn(D), PDiagMat(ones(D))) for i in 1:N]

hmm_diagnorm = HMM(p, A, dists_diagnorm)
hmm_diagnorm_init = HMM(p, A, dists_diagnorm_init)

test_type_stability(hmm_diagnorm, hmm_diagnorm_init; T=100)
