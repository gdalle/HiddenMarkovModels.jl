using Distributions
using HMMBase: HMMBase
using HiddenMarkovModels
using HiddenMarkovModels.HMMTest
using SimpleUnPack
using Test

function test_correctness(hmm, hmm_init; T)
    obs_seq1 = rand(hmm, T).obs_seq
    obs_seq2 = rand(hmm, T).obs_seq
    obs_mat1 = collect(reduce(hcat, obs_seq1)')
    obs_mat2 = collect(reduce(hcat, obs_seq2)')

    nb_seqs = 2
    obs_seqs = [obs_seq1, obs_seq2]

    hmm_base = HMMBase.HMM(deepcopy(hmm))
    hmm_init_base = HMMBase.HMM(deepcopy(hmm_init))

    @testset "Logdensity" begin
        logL1_base = HMMBase.forward(hmm_base, obs_mat1)[2]
        logL2_base = HMMBase.forward(hmm_base, obs_mat2)[2]
        logL = logdensityof(hmm, obs_seqs, nb_seqs)
        @test logL ≈ logL1_base + logL2_base
    end

    @testset "Forward" begin
        (α1_base, logL1_base), (α2_base, logL2_base) = [
            HMMBase.forward(hmm_base, obs_mat1), HMMBase.forward(hmm_base, obs_mat2)
        ]
        (α1, logL1), (α2, logL2) = forward(hmm, obs_seqs, nb_seqs)
        @test isapprox(α1, α1_base[end, :])
        @test isapprox(α2, α2_base[end, :])
        @test logL1 ≈ logL1_base
        @test logL2 ≈ logL2_base
    end

    @testset "Viterbi" begin
        q1_base = HMMBase.viterbi(hmm_base, obs_mat1)
        q2_base = HMMBase.viterbi(hmm_base, obs_mat2)
        q1, q2 = viterbi(hmm, obs_seqs, nb_seqs)
        # Viterbi decoding can vary in case of (infrequent) ties
        @test mean(q1 .== q1_base) > 0.9
        @test mean(q2 .== q2_base) > 0.9
    end

    @testset "Forward-backward" begin
        γ1_base = HMMBase.posteriors(hmm_base, obs_mat1)
        γ2_base = HMMBase.posteriors(hmm_base, obs_mat2)
        (γ1, _), (γ2, _) = forward_backward(hmm, obs_seqs, nb_seqs)
        @test isapprox(γ1, γ1_base')
        @test isapprox(γ2, γ2_base')
    end

    @testset "Baum-Welch" begin
        hmm_est_base, hist_base = HMMBase.fit_mle(
            hmm_init_base, obs_mat1; maxiter=10, tol=-Inf
        )
        logL_evolution_base = hist_base.logtots
        hmm_est, logL_evolution = baum_welch(
            hmm_init, [obs_seq1, obs_seq1], 2; max_iterations=10, atol=-Inf
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
