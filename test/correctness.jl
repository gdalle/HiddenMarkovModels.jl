using Distributions
using HMMBase: HMMBase
using HiddenMarkovModels
using HiddenMarkovModels:
    LightDiagNormal, LightCategorical, logdensityof_with_states, similar_hmms
using LinearAlgebra
using Random
using SimpleUnPack
using SparseArrays
using Test

Random.seed!(63)

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
        @test logL ≈ logdensityof_with_states(hmm, obs_seq, q)
        @test logL >= logdensityof_with_states(hmm, obs_seq, state_seq)
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
        obs_seqs = MultiSeq([copy(obs_seq), copy(obs_seq)])
        hmm_est, logL_evolution = baum_welch(
            hmm_guess, obs_seqs; max_iterations=10, atol=-Inf
        )
        @test isapprox(
            logL_evolution[(begin + 1):end], 2 * logL_evolution_base[begin:(end - 1)]
        )
        @test similar_hmms(hmm_est, HMM(hmm_est_base); atol=1e-5)
    end
end

function test_correctness_baum_welch(
    hmm::AbstractHMM, hmm_guess::AbstractHMM; T::Integer, nb_seqs::Integer, atol
)
    obs_seqs = MultiSeq([rand(hmm, T).obs_seq for _ in 1:nb_seqs])
    hmm_est, logL_evolution = baum_welch(hmm_guess, obs_seqs)
    @test last(logL_evolution) > first(logL_evolution)
    @test similar_hmms(hmm_est, hmm; atol)
end

## Distributions

@testset "Categorical" begin
    init = [0.4, 0.6]
    init_guess = [0.5, 0.5]

    trans = [0.8 0.2; 0.2 0.8]
    trans_guess = [0.7 0.3; 0.3 0.7]

    dists = [Categorical([0.2, 0.8]), Categorical([0.8, 0.2])]
    dists_guess = [Categorical([0.3, 0.7]), Categorical([0.7, 0.3])]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_correctness_baum_welch(hmm, hmm_guess; T=100, nb_seqs=20, atol=0.05)
    test_comparison_hmmbase(hmm, hmm_guess; T=100)
end

@testset "Normal" begin
    init = [0.4, 0.6]
    init_guess = [0.5, 0.5]

    trans = [0.8 0.2; 0.2 0.8]
    trans_guess = [0.7 0.3; 0.3 0.7]

    dists = [Normal(-1), Normal(+1)]
    dists_guess = [Normal(-0.5), Normal(+0.5)]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_correctness_baum_welch(hmm, hmm_guess; T=100, nb_seqs=20, atol=0.05)
    test_comparison_hmmbase(hmm, hmm_guess; T=100)
end

@testset "DiagNormal" begin
    init = [0.4, 0.6]
    init_guess = [0.5, 0.5]

    trans = [0.8 0.2; 0.2 0.8]
    trans_guess = [0.7 0.3; 0.3 0.7]

    D = 3
    dists = [MvNormal(-ones(D), I), MvNormal(+ones(D), I)]
    dists_guess = [MvNormal(-ones(D) / 2, I), MvNormal(+ones(D) / 2, I)]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_correctness_baum_welch(hmm, hmm_guess; T=100, nb_seqs=20, atol=0.05)
    test_comparison_hmmbase(hmm, hmm_guess; T=100)
end

## Light distributions

@testset "LightCategorical" begin
    init = [0.4, 0.6]
    init_guess = [0.5, 0.5]

    trans = [0.8 0.2; 0.2 0.8]
    trans_guess = [0.7 0.3; 0.3 0.7]

    dists = [LightCategorical([0.2, 0.8]), LightCategorical([0.8, 0.2])]
    dists_guess = [LightCategorical([0.3, 0.7]), LightCategorical([0.7, 0.3])]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_correctness_baum_welch(hmm, hmm_guess; T=100, nb_seqs=20, atol=0.05)
end

@testset "LightDiagNormal" begin
    init = [0.4, 0.6]
    init_guess = [0.5, 0.5]

    trans = [0.8 0.2; 0.2 0.8]
    trans_guess = [0.7 0.3; 0.3 0.7]

    D = 3
    dists = [LightDiagNormal(-ones(D), ones(D)), LightDiagNormal(+ones(D), ones(D))]
    dists_guess = [
        LightDiagNormal(-ones(D) / 2, ones(D)), LightDiagNormal(+ones(D) / 2, ones(D))
    ]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_correctness_baum_welch(hmm, hmm_guess; T=100, nb_seqs=20, atol=0.05)
end

## Weird arrays

@testset "Normal sparse" begin
    init = [0.2, 0.6, 0.2]
    trans = sparse([
        0.8 0.2 0.0
        0.0 0.8 0.2
        0.2 0.0 0.8
    ])
    dists = [Normal(-2), Normal(0), Normal(+2)]
    hmm = HMM(init, trans, dists)

    init_guess = [0.3, 0.4, 0.4]
    trans_guess = sparse([
        0.6 0.4 0.0
        0.0 0.6 0.4
        0.4 0.0 0.6
    ])
    dists_guess = [Normal(-1), Normal(0), Normal(+1)]
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    test_correctness_baum_welch(hmm, hmm_guess; T=100, nb_seqs=20, atol=0.05)
end

# Periodic
