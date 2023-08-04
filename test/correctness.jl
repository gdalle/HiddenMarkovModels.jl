using Distributions
using Distributions: PDiagMat
using HMMBase: HMMBase
using HiddenMarkovModels
using SimpleUnPack
using Test

function test_correctness(hmm, hmm_init; T)
    @unpack state_seq, obs_seq = rand(hmm, T)
    obs_mat = collect(reduce(hcat, obs_seq)')

    hmm_base = HMMBase.HMM(deepcopy(hmm))
    hmm_init_base = HMMBase.HMM(deepcopy(hmm_init))

    @testset "Logdensity" begin
        _, logL_base = HMMBase.forward(hmm_base, obs_mat)
        logL = logdensityof(hmm, obs_seq)
        @test logL ≈ logL_base
    end

    @testset "Viterbi" begin
        best_state_seq_base = HMMBase.viterbi(hmm_base, obs_mat)
        best_state_seq = @inferred viterbi(hmm, obs_seq)
        @test isequal(best_state_seq, best_state_seq_base)
    end

    @testset "Forward-backward" begin
        γ_base = HMMBase.posteriors(hmm_base, obs_mat)
        fb = @inferred forward_backward(hmm, obs_seq)
        @test isapprox(fb.γ, γ_base')
    end

    @testset "Baum-Welch" begin
        hmm_est_base, hist_base = HMMBase.fit_mle(
            hmm_init_base, obs_mat; maxiter=10, tol=-Inf
        )
        logL_evolution_base = hist_base.logtots
        hmm_est, logL_evolution = @inferred baum_welch(
            hmm_init, obs_seq; max_iterations=10, atol=-Inf
        )
        @test isapprox(
            logL_evolution[(begin + 1):end], logL_evolution_base[begin:(end - 1)]
        )
        @test isapprox(initial_distribution(hmm_est), hmm_est_base.a)
        @test isapprox(transition_matrix(hmm_est), hmm_est_base.A)
    end
end

N = 5
D = 3
T = 100

p = rand_prob_vec(N)
p_init = rand_prob_vec(N)

A = rand_trans_mat(N)
A_init = rand_trans_mat(N)

# Normal

dists_norm = [Normal(randn(), 1.0) for i in 1:N]
dists_norm_init = [Normal(randn(), 1) for i in 1:N]

hmm_norm = HMM(p, A, dists_norm)
hmm_norm_init = HMM(p_init, A_init, dists_norm_init)

@testset verbose = true "Normal" begin
    test_correctness(hmm_norm, hmm_norm_init; T)
end

# DiagNormal

dists_diagnorm = [DiagNormal(randn(D), PDiagMat(ones(D))) for i in 1:N]
dists_diagnorm_init = [DiagNormal(randn(D), PDiagMat(ones(D) .^ 2)) for i in 1:N]

hmm_diagnorm = HMM(p, A, dists_diagnorm)
hmm_diagnorm_init = HMM(p, A, dists_diagnorm_init)

@testset verbose = true "DiagNormal" begin
    test_correctness(hmm_diagnorm, hmm_diagnorm_init; T)
end
