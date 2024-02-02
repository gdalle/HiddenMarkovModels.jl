using Distributions
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using HiddenMarkovModels: LightDiagNormal, LightCategorical
using HMMTest
using LinearAlgebra
using Random: Random, AbstractRNG, default_rng, seed!
using SparseArrays
using StableRNGs
using Test

rng = StableRNG(63)

## Settings

T, K = 100, 200

init = [0.4, 0.6]
init_guess = [0.5, 0.5]

trans = [0.8 0.2; 0.2 0.8]
trans_guess = [0.7 0.3; 0.3 0.7]

p = [[0.8, 0.2], [0.2, 0.8]]
p_guess = [[0.7, 0.3], [0.3, 0.7]]

μ = [-ones(2), +ones(2)]
μ_guess = [-0.7 * ones(2), +0.7 * ones(2)]

σ = ones(2)

control_seqs = [fill(nothing, rand(rng, T:(2T))) for k in 1:K];
control_seq = reduce(vcat, control_seqs);
seq_ends = cumsum(length.(control_seqs));

## Uncontrolled

@testset "Normal" begin
    dists = [Normal(μ[1][1]), Normal(μ[2][1])]
    dists_guess = [Normal(μ_guess[1][1]), Normal(μ_guess[2][1])]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_identical_hmmbase(rng, hmm, hmm_guess; T)
    test_coherent_algorithms(
        rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.05, init=false
    )
    test_type_stability(rng, hmm, hmm_guess; control_seq, seq_ends)
    test_allocations(rng, hmm, hmm_guess; control_seq, seq_ends)
end

@testset "DiagNormal" begin
    dists = [MvNormal(μ[1], Diagonal(abs2.(σ))), MvNormal(μ[2], Diagonal(abs2.(σ)))]
    dists_guess = [
        MvNormal(μ_guess[1], Diagonal(abs2.(σ))), MvNormal(μ_guess[2], Diagonal(abs2.(σ)))
    ]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_identical_hmmbase(rng, hmm, hmm_guess; T)
    test_coherent_algorithms(
        rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.05, init=false
    )
    test_type_stability(rng, hmm, hmm_guess; control_seq, seq_ends)
end

@testset "LightCategorical" begin
    dists = [LightCategorical(p[1]), LightCategorical(p[2])]
    dists_guess = [LightCategorical(p_guess[1]), LightCategorical(p_guess[2])]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_coherent_algorithms(
        rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.05, init=false
    )
    test_type_stability(rng, hmm, hmm_guess; control_seq, seq_ends)
    test_allocations(rng, hmm, hmm_guess; control_seq, seq_ends)
end

@test_skip @testset "LightDiagNormal" begin
    dists = [LightDiagNormal(μ[1], σ), LightDiagNormal(μ[2], σ)]
    dists_guess = [LightDiagNormal(μ_guess[1], σ), LightDiagNormal(μ_guess[2], σ)]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_coherent_algorithms(
        rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.05, init=false
    )
    test_type_stability(rng, hmm, hmm_guess; control_seq, seq_ends)
    test_allocations(rng, hmm, hmm_guess; control_seq, seq_ends)
end

# Controlled

struct DiffusionHMM{R1,R2,R3} <: AbstractHMM
    init::Vector{R1}
    trans::Matrix{R2}
    means::Vector{R3}
end

HMMs.initialization(hmm::DiffusionHMM) = hmm.init

function HMMs.transition_matrix(hmm::DiffusionHMM, λ::Number)
    @assert 0 <= λ <= 1
    N = length(hmm)
    return (1 - λ) * hmm.trans + λ * ones(N, N) / N
end

function HMMs.obs_distributions(hmm::DiffusionHMM, λ::Number)
    @assert 0 <= λ <= 1
    return [Normal((1 - λ) * hmm.means[i]) for i in 1:length(hmm)]
end

@testset "Controlled" begin
    means = randn(rng, 2)
    hmm = DiffusionHMM(init, trans, means)

    control_seqs = [[rand(rng) for t in 1:rand(T:(2T))] for k in 1:K]
    control_seq = reduce(vcat, control_seqs)
    seq_ends = cumsum(length.(control_seqs))

    test_coherent_algorithms(rng, hmm; control_seq, seq_ends, atol=0.05, init=false)
    test_type_stability(rng, hmm; control_seq, seq_ends)
end
