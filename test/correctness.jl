using Distributions: DiagNormal, PDiagMat
using HMMBase: HMMBase
using HiddenMarkovModels
using Test

N = 5
D = 2

sp = StandardStateProcess(rand_prob_vec(N), rand_trans_mat(N))
op = StandardObservationProcess([DiagNormal(randn(D), PDiagMat(ones(D))) for i in 1:N])
hmm = HMM(sp, op)
hmm_base = HMMBase.HMM(deepcopy(hmm));

sp_init = StandardStateProcess(rand_prob_vec(N), rand_trans_mat(N))
op_init = StandardObservationProcess([DiagNormal(randn(D), PDiagMat(ones(D))) for i in 1:N])
hmm_init = HMM(sp_init, op_init)
hmm_init_base = HMMBase.HMM(deepcopy(hmm_init));

(; state_seq, obs_seq) = rand(hmm, 100);
obs_mat = reduce(hcat, obs_seq)';

@testset "Logdensity" begin
    logL = logdensityof(hmm, obs_seq, NormalScale())
    logL_log = logdensityof(hmm, obs_seq, LogScale())
    _, logL_base = HMMBase.forward(hmm_base, obs_mat)
    @test logL ≈ logL_base
    @test logL_log ≈ logL_base
end

@testset "Viterbi" begin
    best_state_seq = @inferred viterbi(hmm, obs_seq, NormalScale())
    best_state_seq_log = @inferred viterbi(hmm, obs_seq, LogScale())
    best_state_seq_base = HMMBase.viterbi(hmm_base, obs_mat)
    @test isequal(best_state_seq, best_state_seq_base)
    @test isequal(best_state_seq_log, best_state_seq_base)
end

@testset "Forward-backward" begin
    fb = @inferred forward_backward(hmm, obs_seq, NormalScale())
    fb_log = @inferred forward_backward(hmm, obs_seq, LogScale())
    γ_base = HMMBase.posteriors(hmm_base, obs_mat)
    @test isapprox(fb.γ, γ_base')
    @test isapprox(fb_log.logγ, log.(γ_base)')
end

@testset "Baum-Welch" begin
    hmm_est, logL_evolution = @inferred baum_welch(
        copy(hmm_init), [obs_seq], NormalScale(); max_iterations=100, rtol=NaN
    )
    hmm_est_log, logL_evolution_log = @inferred baum_welch(
        copy(hmm_init), [obs_seq], LogScale(); max_iterations=100, rtol=NaN
    )

    hmm_est_base, hist_base = HMMBase.fit_mle(hmm_init_base, obs_mat; maxiter=100, tol=NaN)
    logL_evolution_base = hist_base.logtots

    @test isapprox(logL_evolution[(begin + 1):end], logL_evolution_base[begin:(end - 1)])
    @test isapprox(
        logL_evolution_log[(begin + 1):end], logL_evolution_base[begin:(end - 1)]
    )
    @test isapprox(initial_distribution(hmm_est.state_process), hmm_est_base.a)
    @test isapprox(transition_matrix(hmm_est.state_process), hmm_est_base.A)
    @test isapprox(initial_distribution(hmm_est_log.state_process), hmm_est_base.a)
    @test isapprox(transition_matrix(hmm_est_log.state_process), hmm_est_base.A)
end
