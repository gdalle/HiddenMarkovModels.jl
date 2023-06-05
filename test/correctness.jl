using Distributions: Normal
using HiddenMarkovModels
using HiddenMarkovModels: rand_prob_vec, rand_trans_mat
using HMMBase: HMMBase
using Test

N = 5

# True model

p = rand_prob_vec(N);
A = rand_trans_mat(N);

transitions = MarkovTransitions(p, A);
emissions = VectorEmissions([Normal(float(i), 1.0) for i in 1:N]);

hmm = HMM(transitions, emissions);
hmm_base = HMMBase.HMM(deepcopy(hmm));

# Simulation

(; state_seq, obs_seq) = rand(hmm, 100);

# Inference

logB = HMMs.loglikelihoods(hmm, obs_seq);
logB_base = HMMBase.loglikelihoods(hmm_base, obs_seq)';
@test isapprox(logB, logB_base)

best_state_seq = viterbi(hmm, obs_seq);
best_state_seq_base = HMMBase.viterbi(hmm_base, obs_seq);
@test isequal(best_state_seq, best_state_seq_base)

fb, logL = forward_backward(hmm, obs_seq);
α_base, logL_base = HMMBase.forward(hmm_base, obs_seq);
β_base, _ = HMMBase.backward(hmm_base, obs_seq);
γ_base = HMMBase.posteriors(hmm_base, obs_seq);
@test logL ≈ logL_base
@test isapprox(fb.α, α_base')
@test isapprox(fb.γ, γ_base')

# Learning

p_est = rand_prob_vec(N);
A_est = rand_trans_mat(N);

transitions_est = MarkovTransitions(p_est, A_est);
emissions_est = VectorEmissions([Normal(rand(), 1.0) for i in 1:N]);

hmm_est = HMM(transitions_est, emissions_est);
hmm_init_base = HMMBase.HMM(deepcopy(hmm_est));

logL_evolution = baum_welch!(hmm_est, [obs_seq]; max_iterations=100, rtol=NaN);
hmm_est_base, hist = HMMBase.fit_mle(hmm_init_base, obs_seq; maxiter=100, tol=NaN);
logL_evolution_base = hist.logtots;
@test isapprox(logL_evolution[(begin + 1):end], logL_evolution_base[begin:(end - 1)])
@test isapprox(initial_distribution(hmm_est), hmm_est_base.a)
@test isapprox(transition_matrix(hmm_est), hmm_est_base.A)
