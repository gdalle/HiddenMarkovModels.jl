using Distributions: DiagNormal, PDiagMat
using HMMBase: HMMBase
using HiddenMarkovModels
using HiddenMarkovModels: rand_prob_vec, rand_trans_mat
using LinearAlgebra: Diagonal
using Test: @testset, @test

N = 5
D = 2

# True model

sp = StandardStateProcess(rand_prob_vec(N), rand_trans_mat(N))
op = StandardObservationProcess([DiagNormal(randn(D), PDiagMat(ones(D))) for i in 1:N])
hmm = HMM(sp, op)
hmm_base = HMMBase.HMM(deepcopy(hmm));

# Simulation

(; state_seq, obs_seq) = rand(hmm, 100);
obs_mat = reduce(hcat, obs_seq)';

# Inference

logB = HMMs.loglikelihoods(hmm.obs_process, obs_seq);
logB_base = HMMBase.loglikelihoods(hmm_base, obs_mat)';

@testset "Observation likelihoods" begin
    @test isapprox(logB, logB_base)
end

logL = logdensityof(hmm, obs_seq);
_, logL_base = HMMBase.forward(hmm_base, obs_mat);

@testset "Sequence likelihood" begin
    @test logL ≈ logL_base
end

best_state_seq = @inferred viterbi(hmm, obs_seq);
best_state_seq_base = HMMBase.viterbi(hmm_base, obs_mat);

@testset "Viterbi" begin
    @test isequal(best_state_seq, best_state_seq_base)
end

fb = @inferred forward_backward(hmm, obs_seq);
α_base, _ = HMMBase.forward(hmm_base, obs_mat);
β_base, _ = HMMBase.backward(hmm_base, obs_mat);
γ_base = HMMBase.posteriors(hmm_base, obs_mat);

@testset "Forward-backward" begin
    @test isapprox(fb.α, α_base')
    @test isapprox(fb.γ, γ_base')
end

# Learning

sp_init = StandardStateProcess(rand_prob_vec(N), rand_trans_mat(N))
op_init = StandardObservationProcess([DiagNormal(randn(D), PDiagMat(ones(D))) for i in 1:N])
hmm_init = HMM(sp_init, op_init)
hmm_init_base = HMMBase.HMM(deepcopy(hmm_init));

hmm_est, logL_evolution = @inferred baum_welch(
    hmm_init, [obs_seq]; max_iterations=100, rtol=NaN
);

hmm_est_base, hist_base = HMMBase.fit_mle(hmm_init_base, obs_mat; maxiter=100, tol=NaN);
logL_evolution_base = hist_base.logtots;

@testset "Baum-Welch" begin
    @test isapprox(logL_evolution[(begin + 1):end], logL_evolution_base[begin:(end - 1)])
    @test isapprox(initial_distribution(hmm_est.state_process), hmm_est_base.a)
    @test isapprox(transition_matrix(hmm_est.state_process), hmm_est_base.A)
end
