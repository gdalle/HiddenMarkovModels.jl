using Combinatorics
using Distributions
using Distributions: PDiagMat
using HMMBase: HMMBase
using HiddenMarkovModels
using HiddenMarkovModels: LightDiagNormal, LightCategorical
using LinearAlgebra
using Random
using SimpleUnPack
using SparseArrays
using Test

Random.seed!(63)

function Base.isapprox(hmm::HMM, hmm_base::HMMBase.HMM)
    isapprox(hmm.init, hmm_base.a) || return false
    isapprox(hmm.trans, hmm_base.A) || return false
    for (dist, dist_base) in zip(hmm.dists, hmm_base.B)
        if hasfield(typeof(dist), :μ)
            isapprox(dist.μ, dist_base.μ) || return false
        elseif hasfield(typeof(dist), :p)
            isapprox(dist.p, dist_base.p) || return false
        end
    end
    return true
end

function Base.isapprox(hmm1::AbstractHMM, hmm2::AbstractHMM; atol)
    init1 = initialization(hmm1)
    init2 = initialization(hmm2)
    maximum(abs, init1 - init2) < atol || return false

    trans1 = transition_matrix(hmm1, 1)
    trans2 = transition_matrix(hmm2, 1)
    maximum(abs, trans1 - trans2) < atol || return false

    dists1 = obs_distributions(hmm1, 1)
    dists2 = obs_distributions(hmm2, 1)
    for (dist1, dist2) in zip(dists1, dists2)
        if hasfield(typeof(dist1), :μ)
            maximum(abs, dist1.μ - dist2.μ) < atol || return false
        elseif hasfield(typeof(dist1), :p)
            maximum(abs, dist1.p - dist2.p) < atol || return false
        end
    end
    return true
end

function test_comparison_hmmbase(hmm::AbstractHMM, hmm_guess::AbstractHMM; T::Integer)
    state_seq, obs_seq = rand(hmm, T)
    obs_mat = collect(reduce(hcat, obs_seq)')

    hmm_base = HMMBase.HMM(deepcopy(hmm))
    hmm_guess_base = HMMBase.HMM(deepcopy(hmm_guess))

    @testset "Logdensity" begin
        logL_base = HMMBase.forward(hmm_base, obs_mat)[2]
        logL = logdensityof(hmm, obs_seq)
        @test logL ≈ logL_base
    end

    @testset "Forward" begin
        α_base, logL_base = HMMBase.forward(hmm_base, obs_mat)
        α, logL = forward(hmm, obs_seq)
        @test isapprox(α, α_base[end, :])
        @test logL ≈ logL_base
    end

    @testset "Viterbi" begin
        q_base = HMMBase.viterbi(hmm_base, obs_mat)
        q, logL = viterbi(hmm, obs_seq)
        @test logL ≈ logdensityof(hmm, obs_seq, q)
        # Viterbi decoding can vary in case of (infrequent) ties
        @test mean(q .== q_base) > 0.9
    end

    @testset "Forward-backward" begin
        γ_base = HMMBase.posteriors(hmm_base, obs_mat)
        γ, logL = forward_backward(hmm, obs_seq)
        @test isapprox(γ, γ_base')
    end

    @testset "Baum-Welch" begin
        hmm_est_base, hist_base = HMMBase.fit_mle(
            hmm_guess_base, obs_mat; maxiter=10, tol=-Inf
        )
        logL_evolution_base = hist_base.logtots
        hmm_est, logL_evolution = baum_welch(
            hmm_guess, [obs_seq, obs_seq], 2; max_iterations=10, atol=-Inf
        )
        @test isapprox(
            logL_evolution[(begin + 1):end], 2 * logL_evolution_base[begin:(end - 1)]
        )
        @test isapprox(hmm_est, hmm_est_base)
    end
end

function test_correctness_baum_welch(
    hmm::AbstractHMM, hmm_guess::AbstractHMM; T::Integer, nb_seqs::Integer, atol
)
    obs_seqs = [rand(hmm, T).obs_seq for _ in 1:nb_seqs]
    hmm_est, logL_evolution = baum_welch(hmm_guess, obs_seqs, nb_seqs)
    success_by_perm = Bool[]
    for perm in permutations(1:length(hmm))
        permuted_hmm_est = PermutedHMM(hmm_est, perm)
        push!(success_by_perm, isapprox(permuted_hmm_est, hmm; atol))
    end
    @test last(logL_evolution) > first(logL_evolution)
    @test sum(success_by_perm) == 1
end

## Distributions

@testset "Categorical" begin
    N = 2

    init = [0.3, 0.7]
    trans = [0.8 0.2; 0.2 0.8]
    dists = [Categorical([0.2, 0.8]), Categorical([0.8, 0.2])]
    hmm = HMM(init, trans, dists)

    init_guess = ones(N) / N
    trans_guess = ones(N, N) / N
    dists_guess = [Categorical([0.3, 0.7]), Categorical([0.7, 0.3])]
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_correctness_baum_welch(hmm, hmm_guess; T=200, nb_seqs=100, atol=0.1)
    test_comparison_hmmbase(hmm, hmm_guess; T=100)
end

@testset "Normal" begin
    N = 2

    init = [0.3, 0.7]
    trans = [0.8 0.2; 0.2 0.8]
    dists = [Normal(randn(), 1) for i in 1:N]
    hmm = HMM(init, trans, dists)

    init_guess = ones(N) / N
    trans_guess = ones(N, N) / N
    dists_guess = [Normal(randn(), 1) for i in 1:N]
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_correctness_baum_welch(hmm, hmm_guess; T=100, nb_seqs=100, atol=0.1)
    test_comparison_hmmbase(hmm, hmm_guess; T=100)
end

@testset "DiagNormal" begin
    N, D = 2, 2

    init = [0.3, 0.7]
    trans = [0.8 0.2; 0.2 0.8]
    dists = [DiagNormal(randn(D), PDiagMat(ones(D) .^ 2)) for i in 1:N]
    hmm = HMM(init, trans, dists)

    init_guess = ones(N) / N
    trans_guess = ones(N, N) / N
    dists_guess = [DiagNormal(randn(D), PDiagMat(ones(D) .^ 2)) for i in 1:N]
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_correctness_baum_welch(hmm, hmm_guess; T=100, nb_seqs=100, atol=0.1)
    test_comparison_hmmbase(hmm, hmm_guess; T=100)
end

## Light distributions

@testset "LightCategorical" begin
    N = 2

    init = [0.3, 0.7]
    trans = [0.8 0.2; 0.2 0.8]
    dists = [LightCategorical([0.2, 0.8]), LightCategorical([0.8, 0.2])]
    hmm = HMM(init, trans, dists)

    init_guess = ones(N) / N
    trans_guess = ones(N, N) / N
    dists_guess = [LightCategorical([0.3, 0.7]), LightCategorical([0.7, 0.3])]
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_correctness_baum_welch(hmm, hmm_guess; T=200, nb_seqs=100, atol=0.1)
end

@testset "LightDiagNormal" begin
    N, D = 2, 2

    init = [0.3, 0.7]
    trans = [0.8 0.2; 0.2 0.8]
    dists = [LightDiagNormal(randn(D), ones(D)) for i in 1:N]
    hmm = HMM(init, trans, dists)

    init_guess = ones(N) / N
    trans_guess = ones(N, N) / N
    dists_guess = [LightDiagNormal(randn(D), ones(D)) for i in 1:N]
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_correctness_baum_welch(hmm, hmm_guess; T=100, nb_seqs=100, atol=0.1)
end

## Weird arrays

@testset "Normal sparse" begin
    N = 3

    init = [0.3, 0.5, 0.2]
    trans = sparse([
        0.8 0.2 0.0
        0.0 0.8 0.2
        0.2 0.0 0.8
    ])
    dists = [Normal(i, 1) for i in 1:N]
    hmm = HMM(init, trans, dists)

    init_guess = ones(N) / N
    trans_guess = sparse([
        0.5 0.5 0.0
        0.0 0.5 0.5
        0.5 0.0 0.5
    ])
    dists_guess = [Normal(i + 0.3, 1) for i in 1:N]
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_correctness_baum_welch(hmm, hmm_guess; T=100, nb_seqs=100, atol=0.1)
    test_comparison_hmmbase(hmm, hmm_guess; T=100)
end
