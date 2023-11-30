using HiddenMarkovModels: LightCategorical, LightDiagNormal
using Statistics
using Test

function test_fit_allocs(dist, x, w)
    dist_copy = deepcopy(dist)
    allocs = @allocated fit!(dist_copy, x, w)
    @test allocs == 0
end

@testset "LightCategorical" begin
    p = rand_prob_vec(10)
    dist = LightCategorical(p)
    x = [rand(dist) for _ in 1:100_000]
    # Simulation
    val_count = zeros(Int, length(p))
    for k in x
        val_count[k] += 1
    end
    @test val_count ./ length(x) ≈ p atol = 1e-2
    # Fitting
    dist_est = deepcopy(dist)
    w = ones(length(x))
    fit!(dist_est, x, w)
    @test dist_est.p ≈ p atol = 1e-2
    test_fit_allocs(dist, x, w)
end

@testset "LightDiagNormal" begin
    μ = randn(10)
    σ = rand(10)
    dist = LightDiagNormal(μ, σ)
    x = [rand(dist) for _ in 1:100_000]
    # Simulation
    @test mean(x) ≈ μ atol = 1e-2
    @test std(x) ≈ σ atol = 1e-2
    # Fitting
    dist_est = deepcopy(dist)
    w = ones(length(x))
    fit!(dist_est, x, w)
    @test dist_est.μ ≈ μ atol = 1e-2
    @test dist_est.σ ≈ σ atol = 1e-2
    test_fit_allocs(dist, x, w)
end
