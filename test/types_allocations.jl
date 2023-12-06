using Distributions
using HiddenMarkovModels
using HiddenMarkovModels: LightDiagNormal, LightCategorical, rand_prob_vec, rand_trans_mat
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

    @test_opt target_modules = (HMMs,) rand(hmm, T)
    @test_call target_modules = (HMMs,) rand(hmm, T)

    @test_opt target_modules = (HMMs,) logdensityof(hmm, obs_seq)
    @test_call target_modules = (HMMs,) logdensityof(hmm, obs_seq)

    @test_opt target_modules = (HMMs,) forward(hmm, obs_seq)
    @test_call target_modules = (HMMs,) forward(hmm, obs_seq)

    @test_opt target_modules = (HMMs,) viterbi(hmm, obs_seq)
    @test_call target_modules = (HMMs,) viterbi(hmm, obs_seq)

    @test_opt target_modules = (HMMs,) forward_backward(hmm, obs_seq)
    @test_call target_modules = (HMMs,) forward_backward(hmm, obs_seq)

    @test_opt target_modules = (HMMs,) baum_welch(hmm, obs_seq; max_iterations=1)
    @test_call target_modules = (HMMs,) baum_welch(hmm, obs_seq; max_iterations=1)
end

function test_allocations(hmm::AbstractHMM; T::Integer)
    obs_seq = rand(hmm, T).obs_seq
    obs_seqs = MultiSeq([rand(hmm, T).obs_seq for _ in 1:2])

    ## Forward
    forward(hmm, obs_seq)  # compile
    f_storage = HMMs.initialize_forward(hmm, obs_seq)
    allocs = @allocated HiddenMarkovModels.forward!(f_storage, hmm, obs_seq)
    @test allocs == 0

    ## Viterbi
    viterbi(hmm, obs_seq)  # compile
    v_storage = HMMs.initialize_viterbi(hmm, obs_seq)
    allocs = @allocated HMMs.viterbi!(v_storage, hmm, obs_seq)
    @test allocs == 0

    ## Forward-backward
    forward_backward(hmm, obs_seq)  # compile
    fb_storage = HMMs.initialize_forward_backward(hmm, obs_seq)
    allocs = @allocated HMMs.forward_backward!(fb_storage, hmm, obs_seq)
    @test allocs == 0

    ## Baum-Welch
    baum_welch(hmm, obs_seqs)  # compile
    bw_storage = HMMs.initialize_baum_welch(hmm, obs_seqs; max_iterations=1)
    hmm_guess = deepcopy(hmm)
    allocs = @allocated HMMs.baum_welch!(
        bw_storage,
        hmm_guess,
        obs_seqs;
        atol=-Inf,
        max_iterations=1,
        loglikelihood_increasing=false,
    )
    @test allocs == 0
end

N, D, T, nb_seqs, R = 3, 2, 10, 5, Float32

## Distributions

@testset "Normal" begin
    init = rand_prob_vec(R, N)
    trans = rand_trans_mat(R, N)
    dists = [Normal(randn(R), 1) for i in 1:N]
    hmm = HMM(init, trans, dists)
    test_type_stability(hmm; T)
    test_allocations(hmm; T)
end

@testset "DiagNormal" begin
    init = rand_prob_vec(R, N)
    trans = rand_trans_mat(R, N)
    dists = [MvNormal(randn(R, D), I) for i in 1:N]
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
    @test_skip test_allocations(hmm; T)
end

@testset "LightDiagNormal" begin
    init = rand_prob_vec(R, N)
    trans = rand_trans_mat(R, N)
    dists = [LightDiagNormal(randn(R, D), ones(D)) for i in 1:N]
    hmm = HMM(init, trans, dists)
    test_type_stability(hmm; T)
    test_allocations(hmm; T)
end

## Weird arrays

@testset "Normal sparse" begin
    init = rand_prob_vec(R, N)
    trans = sparse(SymTridiagonal(rand(R, N), rand(R, N - 1)))
    foreach(HMMs.sum_to_one!, eachrow(trans))
    dists = [Normal(randn(R), 1) for i in 1:N]
    hmm = HMM(init, trans, dists)
    test_type_stability(hmm; T)
    test_allocations(hmm; T)
end
