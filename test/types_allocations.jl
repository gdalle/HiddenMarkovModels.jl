using Distributions
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using HiddenMarkovModels: LightDiagNormal, LightCategorical, rand_prob_vec, rand_trans_mat
using JET: @test_opt, @test_call
using LinearAlgebra
using Random: AbstractRNG, default_rng, seed!
using SimpleUnPack
using SparseArrays
using Test

rng = default_rng()
seed!(rng, 63)

function test_type_stability(
    rng::AbstractRNG,
    hmm::AbstractHMM;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int}=[length(control_seq)],
)
    state_seq, obs_seq = rand(rng, hmm, control_seq)

    @test_opt target_modules = (HMMs,) rand(hmm, control_seq)
    @test_call target_modules = (HMMs,) rand(hmm, control_seq)

    @test_opt target_modules = (HMMs,) logdensityof(hmm, obs_seq; control_seq, seq_ends)
    @test_call target_modules = (HMMs,) logdensityof(hmm, obs_seq; control_seq, seq_ends)
    @test_opt target_modules = (HMMs,) logdensityof(
        hmm, obs_seq, state_seq; control_seq, seq_ends
    )
    @test_call target_modules = (HMMs,) logdensityof(
        hmm, obs_seq, state_seq; control_seq, seq_ends
    )

    @test_opt target_modules = (HMMs,) forward(hmm, obs_seq; control_seq, seq_ends)
    @test_call target_modules = (HMMs,) forward(hmm, obs_seq; control_seq, seq_ends)

    @test_opt target_modules = (HMMs,) viterbi(hmm, obs_seq; control_seq, seq_ends)
    @test_call target_modules = (HMMs,) viterbi(hmm, obs_seq; control_seq, seq_ends)

    @test_opt target_modules = (HMMs,) forward_backward(hmm, obs_seq; control_seq, seq_ends)
    @test_call target_modules = (HMMs,) forward_backward(
        hmm, obs_seq; control_seq, seq_ends
    )

    @test_opt target_modules = (HMMs,) baum_welch(
        hmm, obs_seq; control_seq, seq_ends, max_iterations=1
    )
    @test_call target_modules = (HMMs,) baum_welch(
        hmm, obs_seq; control_seq, seq_ends, max_iterations=1
    )
end

function test_allocations(
    rng::AbstractRNG,
    hmm::AbstractHMM;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
)
    obs_seq = mapreduce(vcat, eachindex(seq_ends)) do k
        t1, t2 = HMMs.seq_limits(seq_ends, k)
        rand(rng, hmm, control_seq[t1:t2]).obs_seq
    end

    ## Forward
    forward(hmm, obs_seq; control_seq, seq_ends)  # compile
    f_storage = HMMs.initialize_forward(hmm, obs_seq; control_seq, seq_ends)
    allocs = @allocated HiddenMarkovModels.forward!(
        f_storage, hmm, obs_seq; control_seq, seq_ends
    )
    @test allocs == 0

    ## Viterbi
    viterbi(hmm, obs_seq; control_seq, seq_ends)  # compile
    v_storage = HMMs.initialize_viterbi(hmm, obs_seq; control_seq, seq_ends)
    allocs = @allocated HMMs.viterbi!(v_storage, hmm, obs_seq; control_seq, seq_ends)
    @test allocs == 0

    ## Forward-backward
    forward_backward(hmm, obs_seq; control_seq, seq_ends)  # compile
    fb_storage = HMMs.initialize_forward_backward(hmm, obs_seq; control_seq, seq_ends)
    allocs = @allocated HMMs.forward_backward!(
        fb_storage, hmm, obs_seq; control_seq, seq_ends
    )
    @test allocs == 0

    ## Baum-Welch
    baum_welch(hmm, obs_seq; control_seq, seq_ends, max_iterations=1)  # compile
    fb_storage = HMMs.initialize_forward_backward(hmm, obs_seq; control_seq, seq_ends)
    logL_evolution = Float64[]
    sizehint!(logL_evolution, 1)
    hmm_guess = deepcopy(hmm)
    allocs = @allocated HMMs.baum_welch!(
        fb_storage,
        logL_evolution,
        hmm_guess,
        obs_seq;
        control_seq,
        seq_ends,
        atol=-Inf,
        max_iterations=1,
        loglikelihood_increasing=false,
    )
    @test allocs == 0
end

N, D, T, K = 3, 2, 10, 4
R = Float64
control_seq = fill(nothing, K * T)
seq_ends = T:T:(K * T)
init = ones(R, N) / N
trans = ones(R, N, N) / N

## Distributions

@testset "Normal" begin
    dists = [Normal(randn(rng, R), 1) for i in 1:N]
    hmm = HMM(init, trans, dists)
    test_type_stability(rng, hmm; control_seq, seq_ends)
    test_allocations(rng, hmm; control_seq, seq_ends)
end

@testset "DiagNormal" begin
    dists = [MvNormal(randn(rng, R, D), I) for i in 1:N]
    hmm = HMM(init, trans, dists)
    test_type_stability(rng, hmm; control_seq, seq_ends)
end

## Light distributions

@testset "LightCategorical" begin
    dists = [LightCategorical(rand_prob_vec(rng, R, D)) for i in 1:N]
    hmm = HMM(init, trans, dists)
    test_type_stability(rng, hmm; control_seq, seq_ends)
    test_allocations(rng, hmm; control_seq, seq_ends)
end

@testset "LightDiagNormal" begin
    dists = [LightDiagNormal(randn(rng, R, D), ones(R, D)) for i in 1:N]
    hmm = HMM(init, trans, dists)
    test_type_stability(rng, hmm; control_seq, seq_ends)
    test_allocations(rng, hmm; control_seq, seq_ends)
end

## Weird arrays

@testset "Normal sparse" begin
    trans_sparse = sparse(SymTridiagonal(rand(rng, R, N), rand(rng, R, N - 1)))
    foreach(HMMs.sum_to_one!, eachrow(trans_sparse))
    dists = [Normal(randn(rng, R), 1) for i in 1:N]
    hmm = HMM(init, trans_sparse, dists)
    test_type_stability(rng, hmm; control_seq, seq_ends)
    test_allocations(rng, hmm; control_seq, seq_ends)
end
