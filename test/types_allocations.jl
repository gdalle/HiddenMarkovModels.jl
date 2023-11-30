using Distributions
using Distributions: PDiagMat
using HiddenMarkovModels
using HiddenMarkovModels: LightDiagNormal, LightCategorical
import HiddenMarkovModels as HMMs
using JET
using LinearAlgebra
using Random
using SimpleUnPack
using SparseArrays
using Test

Random.seed!(63)

function test_type_stability(hmm::AbstractHMM; T::Integer)
    state_seq, obs_seq = rand(hmm, T)

    @testset "Logdensity" begin
        @test_opt target_modules = (HMMs,) logdensityof(hmm, obs_seq, state_seq)
        @test_call target_modules = (HMMs,) logdensityof(hmm, obs_seq, state_seq)
        @test_opt target_modules = (HMMs,) logdensityof(hmm, obs_seq)
        @test_call target_modules = (HMMs,) logdensityof(hmm, obs_seq)
    end

    @testset "Forward" begin
        @test_opt target_modules = (HMMs,) forward(hmm, obs_seq)
        @test_call target_modules = (HMMs,) forward(hmm, obs_seq)
    end

    @testset "Viterbi" begin
        @test_opt target_modules = (HMMs,) viterbi(hmm, obs_seq)
        @test_call target_modules = (HMMs,) viterbi(hmm, obs_seq)
    end

    @testset "Forward-backward" begin
        @test_opt target_modules = (HMMs,) forward_backward(hmm, obs_seq)
        @test_call target_modules = (HMMs,) forward_backward(hmm, obs_seq)
    end

    @testset "Baum-Welch" begin
        @test_opt target_modules = (HMMs,) baum_welch(hmm, obs_seq; max_iterations=1)
        @test_call target_modules = (HMMs,) baum_welch(hmm, obs_seq; max_iterations=1)
    end
end

function test_allocations(hmm::AbstractHMM; T::Integer, nb_seqs::Integer)
    obs_seq = rand(hmm, T).obs_seq
    obs_seqs = [rand(hmm, T).obs_seq for _ in 1:nb_seqs]

    ## Forward
    f_storage = HMMs.initialize_forward(hmm, obs_seq)
    HiddenMarkovModels.forward!(f_storage, hmm, obs_seq)
    allocs = @allocated HiddenMarkovModels.forward!(f_storage, hmm, obs_seq)
    @test allocs == 0

    ## Viterbi
    v_storage = HMMs.initialize_viterbi(hmm, obs_seq)
    HMMs.viterbi!(v_storage, hmm, obs_seq)
    allocs = @allocated HMMs.viterbi!(v_storage, hmm, obs_seq)
    @test allocs == 0

    ## Forward-backward
    fb_storage = HMMs.initialize_forward_backward(hmm, obs_seq)
    HMMs.forward_backward!(fb_storage, hmm, obs_seq)
    allocs = @allocated HMMs.forward_backward!(fb_storage, hmm, obs_seq)
    @test allocs == 0

    ## Baum-Welch
    fb_storages = [
        HMMs.initialize_forward_backward(hmm, obs_seqs[k]) for k in eachindex(obs_seqs)
    ]
    bw_storage = HMMs.initialize_baum_welch(hmm, fb_storages, obs_seqs)
    logL_evolution = HMMs.initialize_logL_evolution(hmm, obs_seqs; max_iterations=1)
    allocs = @allocated HMMs.baum_welch!(
        hmm,
        fb_storages,
        bw_storage,
        logL_evolution,
        obs_seqs;
        atol=-Inf,
        max_iterations=1,
        loglikelihood_increasing=false,
    )
    @test allocs == 0
end

N, D, T, nb_seqs, R = 3, 2, 100, 5, Float32

## Distributions

@testset "Categorical" begin
    init = rand_prob_vec(R, N)
    trans = rand_trans_mat(R, N)
    dists = [Categorical(rand_prob_vec(R, D)) for i in 1:N]
    hmm = HMM(init, trans, dists)
    test_type_stability(hmm; T)
end

@testset "Normal" begin
    init = rand_prob_vec(R, N)
    trans = rand_trans_mat(R, N)
    dists = [Normal(randn(R), 1) for i in 1:N]
    hmm = HMM(init, trans, dists)
    test_type_stability(hmm; T)
    test_allocations(hmm; T, nb_seqs)
end

@testset "DiagNormal" begin
    init = rand_prob_vec(R, N)
    trans = rand_trans_mat(R, N)
    dists = [DiagNormal(randn(R, D), PDiagMat(ones(D) .^ 2)) for i in 1:N]
    hmm = HMM(init, trans, dists)
    test_type_stability(hmm; T)
end

## Light distributions

@testset "LightCategorical" begin
    init = rand_prob_vec(R, N)
    trans = rand_trans_mat(R, N)
    dists = [LightCategorical(rand_prob_vec(R, D)) for i in 1:N]
    hmm = HMM(init, trans, dists)
    test_type_stability(hmm; T)
    @test_skip test_allocations(hmm; T, nb_seqs)
end

@testset "LightDiagNormal" begin
    init = rand_prob_vec(R, N)
    trans = rand_trans_mat(R, N)
    dists = [LightDiagNormal(randn(R, D), ones(D)) for i in 1:N]
    hmm = HMM(init, trans, dists)
    test_type_stability(hmm; T)
    test_allocations(hmm; T, nb_seqs)
end

## Weird arrays

@testset "Normal sparse" begin
    init = rand_prob_vec(R, N)
    trans = sparse(SymTridiagonal(rand(R, N), rand(R, N - 1)))
    foreach(HMMs.sum_to_one!, eachrow(trans))
    dists = [Normal(randn(R), 1) for i in 1:N]
    hmm = HMM(init, trans, dists)
    test_type_stability(hmm; T)
    test_allocations(hmm; T, nb_seqs)
end
