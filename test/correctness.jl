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

T, K = 50, 200

init = [0.4, 0.6]
init_guess = [0.5, 0.5]

trans = [0.7 0.3; 0.3 0.7]
trans_guess = [0.6 0.4; 0.4 0.6]

p = [[0.8, 0.2], [0.2, 0.8]]
p_guess = [[0.7, 0.3], [0.3, 0.7]]

μ = [-ones(2), +ones(2)]
μ_guess = [-0.8 * ones(2), +0.8 * ones(2)]

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

    test_identical_hmmbase(rng, hmm, T; hmm_guess)
    test_coherent_algorithms(rng, hmm, control_seq; seq_ends, hmm_guess, init=false)
    test_type_stability(rng, hmm, control_seq; seq_ends, hmm_guess)
    test_allocations(rng, hmm, control_seq; seq_ends, hmm_guess)
end

@testset "DiagNormal" begin
    dists = [MvNormal(μ[1], Diagonal(abs2.(σ))), MvNormal(μ[2], Diagonal(abs2.(σ)))]
    dists_guess = [
        MvNormal(μ_guess[1], Diagonal(abs2.(σ))), MvNormal(μ_guess[2], Diagonal(abs2.(σ)))
    ]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_identical_hmmbase(rng, hmm, T; hmm_guess)
    test_coherent_algorithms(rng, hmm, control_seq; seq_ends, hmm_guess, init=false)
    test_type_stability(rng, hmm, control_seq; seq_ends, hmm_guess)
end

@testset "LightCategorical" begin
    dists = [LightCategorical(p[1]), LightCategorical(p[2])]
    dists_guess = [LightCategorical(p_guess[1]), LightCategorical(p_guess[2])]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_coherent_algorithms(rng, hmm, control_seq; seq_ends, hmm_guess, init=false)
    test_type_stability(rng, hmm, control_seq; seq_ends, hmm_guess)
    test_allocations(rng, hmm, control_seq; seq_ends, hmm_guess)
end

@testset "LightDiagNormal" begin
    dists = [LightDiagNormal(μ[1], σ), LightDiagNormal(μ[2], σ)]
    dists_guess = [LightDiagNormal(μ_guess[1], σ), LightDiagNormal(μ_guess[2], σ)]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_coherent_algorithms(rng, hmm, control_seq; seq_ends, hmm_guess, init=false)
    test_type_stability(rng, hmm, control_seq; seq_ends, hmm_guess)
    test_allocations(rng, hmm, control_seq; seq_ends, hmm_guess)
end
