using Distributions: Normal
using HiddenMarkovModels
using HiddenMarkovModels: rand_prob_vec, rand_trans_mat
using HMMBase: HMMBase
using Test: @testset, @test

N = 5

# True model

p = rand_prob_vec(N)
A = rand_trans_mat(N)
transitions = StandardTransitions(p, A)
emissions = VectorEmissions([Normal(float(i), 1.0) for i in 1:N])
hmm = HMM(transitions, emissions)
hmm_base = HMMBase.HMM(deepcopy(hmm))

# Simulation

(; state_seq, obs_seq) = rand(hmm, 100)

# Inference

logB = HMMs.loglikelihoods(hmm, obs_seq)
logB_base = HMMBase.loglikelihoods(hmm_base, obs_seq)'

@testset "Likelihoods" begin
    @test isapprox(logB, logB_base)
end

best_state_seq = viterbi(hmm, obs_seq)
best_state_seq_base = HMMBase.viterbi(hmm_base, obs_seq)

@testset "Viterbi" begin
    @test isequal(best_state_seq, best_state_seq_base)
end

fb, logL = forward_backward(hmm, obs_seq)
α_base, logL_base = HMMBase.forward(hmm_base, obs_seq)
β_base, _ = HMMBase.backward(hmm_base, obs_seq)
γ_base = HMMBase.posteriors(hmm_base, obs_seq)

@testset "Forward-backward" begin
    @test logL ≈ logL_base
    @test isapprox(fb.α, α_base')
    @test isapprox(fb.γ, γ_base')
end

# Learning

p_init = rand_prob_vec(N)
A_init = rand_trans_mat(N)
transitions_init = StandardTransitions(p_init, A_init)
emissions_init = VectorEmissions([Normal(rand(), 1.0) for i in 1:N])
hmm_init = HMM(transitions_init, emissions_init)
hmm_init_base = HMMBase.HMM(copy(hmm_init))

hmm_est, logL_evolution = baum_welch(hmm_init, [obs_seq]; max_iterations=100, rtol=NaN)
hmm_est_base, hist_base = HMMBase.fit_mle(hmm_init_base, obs_seq; maxiter=100, tol=NaN)
logL_evolution_base = hist_base.logtots

@testset "Baum-Welch" begin
    @test isapprox(logL_evolution[(begin + 1):end], logL_evolution_base[begin:(end - 1)])
    @test isapprox(initial_distribution(hmm_est), hmm_est_base.a)
    @test isapprox(transition_matrix(hmm_est), hmm_est_base.A)
end
