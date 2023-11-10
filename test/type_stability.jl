using HiddenMarkovModels
using HiddenMarkovModels.Test
import HiddenMarkovModels as HMMs
using JET
using SimpleUnPack
using Test

function test_type_stability(hmm, hmm_init; T)
    obs_seq = rand(hmm, T).obs_seq

    @testset "Logdensity" begin
        @inferred logdensityof(hmm, obs_seq)
        @test_opt target_modules = (HMMs,) logdensityof(hmm, obs_seq)
        @test_call target_modules = (HMMs,) logdensityof(hmm, obs_seq)
    end

    @testset "Forward" begin
        @inferred forward(hmm, obs_seq)
        @test_opt target_modules = (HMMs,) forward(hmm, obs_seq)
        @test_call target_modules = (HMMs,) forward(hmm, obs_seq)
    end

    @testset "Viterbi" begin
        @inferred viterbi(hmm, obs_seq)
        @test_opt target_modules = (HMMs,) viterbi(hmm, obs_seq)
        @test_call target_modules = (HMMs,) viterbi(hmm, obs_seq)
    end

    @testset "Forward-backward" begin
        @inferred forward_backward(hmm, obs_seq)
        @test_opt target_modules = (HMMs,) forward_backward(hmm, obs_seq)
        @test_call target_modules = (HMMs,) forward_backward(hmm, obs_seq)
    end

    @testset "Baum-Welch" begin
        @inferred baum_welch(hmm_init, obs_seq)
        @test_opt target_modules = (HMMs,) baum_welch(hmm_init, obs_seq; max_iterations=2)
        @test_call target_modules = (HMMs,) baum_welch(hmm_init, obs_seq; max_iterations=2)
    end
end

N, D, T = 3, 2, 1000

@testset "Categorical" begin
    test_type_stability(rand_categorical_hmm(N, D), rand_categorical_hmm(N, D); T)
end

@testset "Normal" begin
    test_type_stability(rand_gaussian_hmm_1d(N), rand_gaussian_hmm_1d(N); T)
end

@testset "Normal sparse" begin
    test_type_stability(
        rand_gaussian_hmm_1d(N; sparse_trans=true),
        rand_gaussian_hmm_1d(N; sparse_trans=true);
        T,
    )
end

@testset "DiagNormal" begin
    test_type_stability(rand_gaussian_hmm_2d(N, D), rand_gaussian_hmm_2d(N, D); T)
end

@testset "LightDiagNormal" begin
    test_type_stability(
        rand_gaussian_hmm_2d_light(N, D), rand_gaussian_hmm_2d_light(N, D); T
    )
end
