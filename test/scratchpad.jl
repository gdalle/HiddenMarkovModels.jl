using HiddenMarkovModels
using HiddenMarkovModels: rand_prob_vec, rand_trans_mat, MyNormal
using HiddenMarkovModels: likelihoods, loglikelihoods

N = 3

p = rand_prob_vec(N)
A = rand_trans_mat(N)

transitions = MarkovTransitions(p, A)
emissions = VectorEmissions([MyNormal(float(i), 1.0) for i in 1:N])

hmm = HMM(transitions, emissions)

(; state_seq, obs_seq) = rand(hmm, 1000);
obs_seqs = [rand(hmm, 200).obs_seq for k in 1:10];

likelihoods(hmm, obs_seq)
loglikelihoods(hmm, obs_seq)

forward_backward(hmm, obs_seq)

viterbi(hmm, obs_seq)

p_est = rand_prob_vec(N)
A_est = rand_trans_mat(N)

transitions_est = MarkovTransitions(p_est, A_est)
emissions_est = VectorEmissions([MyNormal(rand(), 1.0) for i in 1:N])

hmm_est = HMM(transitions_est, emissions_est)

baum_welch!(hmm_est, obs_seqs)
