using Distributions
using HiddenMarkovModels:
    LightCategorical, LightDiagNormal, logdensityof, rand_prob_vec, rand_trans_mat
using LinearAlgebra
using Statistics
using StatsAPI: fit!
using StableRNGs
using Test

rng = StableRNG(63)

function test_randprobvec(p)
    @test all(>=(0), p)
    @test sum(p) ≈ 1
end

function test_randtransmat(A)
    foreach(eachrow(A)) do p
        test_randprobvec(p)
    end
end

@testset "Rand prob" begin
    n = 10
    test_randprobvec(rand_prob_vec(n))
    test_randprobvec(rand_prob_vec(rng, n))
    test_randprobvec(rand_prob_vec(Float32, n))
    test_randprobvec(rand_prob_vec(rng, Float32, n))
    test_randtransmat(rand_trans_mat(n))
    test_randtransmat(rand_trans_mat(rng, n))
    test_randtransmat(rand_trans_mat(Float32, n))
    test_randtransmat(rand_trans_mat(rng, Float32, n))
end

function test_fit_allocs(dist, x, w)
    dist_copy = deepcopy(dist)
    allocs = @allocated fit!(dist_copy, x, w)
    @test allocs == 0
end

@testset "LightCategorical" begin
    p = rand_prob_vec(rng, 10)
    dist = LightCategorical(p)
    @test startswith(string(dist), "LightCategorical")
    x = [(@inferred rand(rng, dist)) for _ in 1:100_000]
    # Simulation
    val_count = zeros(Int, length(p))
    for k in x
        val_count[k] += 1
    end
    @test val_count ./ length(x) ≈ p atol = 2e-2
    # Fitting
    dist_est = LightCategorical(rand_prob_vec(rng, 10))
    w = ones(length(x))
    fit!(dist_est, x, w)
    @test dist_est.p ≈ p atol = 2e-2
    test_fit_allocs(dist, x, w)
    # Logdensity
    @test logdensityof(dist, x[1]) ≈ logdensityof(Categorical(p), x[1])
end

@testset "LightDiagNormal" begin
    μ = randn(rng, 10)
    σ = rand(rng, 10)
    dist = LightDiagNormal(μ, σ)
    @test startswith(string(dist), "LightDiagNormal")
    x = [(@inferred rand(rng, dist)) for _ in 1:100_000]
    # Simulation
    @test mean(x) ≈ μ atol = 2e-2
    @test std(x) ≈ σ atol = 2e-2
    # Fitting
    dist_est = LightDiagNormal(randn(rng, 10), rand(rng, 10))
    w = ones(length(x))
    fit!(dist_est, x, w)
    @test dist_est.μ ≈ μ atol = 2e-2
    @test dist_est.σ ≈ σ atol = 2e-2
    test_fit_allocs(dist, x, w)
    # Logdensity
    @test logdensityof(dist, x[1]) ≈ logdensityof(MvNormal(μ, Diagonal(abs2.(σ))), x[1])
end
