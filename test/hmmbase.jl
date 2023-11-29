using Distributions
using Distributions: PDiagMat
using HMMBase: HMMBase
using HiddenMarkovModels
using SimpleUnPack
using Test

function compare_hmmbase(hmm, hmm_init; T)
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
        q, logL = viterbi(hmm, obs_seq)
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
            hmm_init_base, obs_mat; maxiter=10, tol=-Inf
        )
        logL_evolution_base = hist_base.logtots
        hmm_est, logL_evolution = baum_welch(
            hmm_init, [obs_seq, obs_seq], 2; max_iterations=10, atol=-Inf
        )
        @test isapprox(
            logL_evolution[(begin + 1):end], 2 * logL_evolution_base[begin:(end - 1)]
        )
        @test isapprox(hmm_est.init, hmm_est_base.a)
        @test isapprox(hmm_est.trans, hmm_est_base.A)

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
    p = ones(N) / N
    A = rand_trans_mat(N)
    d = [Categorical(rand_prob_vec(D)) for i in 1:N]
    hmm = HMM(p, A, d)

    p_init = ones(N) / N
    A_init = rand_trans_mat(N)
    d_init = [Categorical(rand_prob_vec(D)) for i in 1:N]
    hmm_init = HMM(p_init, A_init, d_init)

    compare_hmmbase(hmm, hmm_init; T)
end

@testset "Normal" begin
    p = ones(N) / N
    A = rand_trans_mat(N)
    d = [Normal(randn(), 1) for i in 1:N]
    hmm = HMM(p, A, d)

    p_init = ones(N) / N
    A_init = rand_trans_mat(N)
    d_init = [Normal(randn(), 1) for i in 1:N]
    hmm_init = HMM(p_init, A_init, d_init)

    compare_hmmbase(hmm, hmm_init; T)
end

@testset "DiagNormal" begin
    p = ones(N) / N
    A = rand_trans_mat(N)
    d = [DiagNormal(randn(D), PDiagMat(ones(D) .^ 2)) for i in 1:N]
    hmm = HMM(p, A, d)

    p_init = ones(N) / N
    A_init = rand_trans_mat(N)
    d_init = [DiagNormal(randn(D), PDiagMat(ones(D) .^ 2)) for i in 1:N]
    hmm_init = HMM(p_init, A_init, d_init)

    compare_hmmbase(hmm, hmm_init; T)
end
