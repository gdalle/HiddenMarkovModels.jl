using Distributions
using HMMBase: HMMBase
using HiddenMarkovModels
using HiddenMarkovModels.HMMTest
using SimpleUnPack
using Test

function test_correctness(hmm, hmm_init; T)
    obs_seq = rand(hmm, T).obs_seq
    obs_mat = collect(reduce(hcat, obs_seq)')

    hmm_base = HMMBase.HMM(deepcopy(hmm))
    hmm_init_base = HMMBase.HMM(deepcopy(hmm_init))

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
        q = viterbi(hmm, obs_seq)
        # Viterbi decoding can vary in case of (infrequent) ties
        @test mean(q .== q_base) > 0.9
    end

    @testset "Forward-backward" begin
        γ_base = HMMBase.posteriors(hmm_base, obs_mat)
        γ, _, _ = forward_backward(hmm, obs_seq)
        @test isapprox(γ, γ_base')
    end

    @testset "Baum-Welch" begin
        hmm_est_base, hist_base = HMMBase.fit_mle(
            hmm_init_base, obs_mat; maxiter=10, tol=-Inf
        )
        logL_evolution_base = hist_base.logtots
        hmm_est, logL_evolution = baum_welch(
            hmm_init, [obs_seq, obs_seq], 2; max_iterations=10, atol=-Inf
        )
        @test isapprox(
            logL_evolution[(begin + 1):end], 2 * logL_evolution_base[begin:(end - 1)]
        )
        @test isapprox(initialization(hmm_est), hmm_est_base.a)
        @test isapprox(transition_matrix(hmm_est), hmm_est_base.A)

        for (dist, dist_base) in zip(hmm.dists, hmm_base.B)
            if hasfield(typeof(dist), :μ)
                @test isapprox(dist.μ, dist_base.μ)
            elseif hasfield(typeof(dist), :p)
                @test isapprox(dist.p, dist_base.p)
            end
        end
    end
end

N, D, T = 3, 2, 100

@testset "Categorical" begin
    test_correctness(rand_categorical_hmm(N, D), rand_categorical_hmm(N, D); T)
end

@testset "Normal" begin
    test_correctness(rand_gaussian_hmm_1d(N), rand_gaussian_hmm_1d(N); T)
end

@testset "DiagNormal" begin
    test_correctness(rand_gaussian_hmm_2d(N, D), rand_gaussian_hmm_2d(N, D); T)
end
