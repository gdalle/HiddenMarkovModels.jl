# # Basics

#=
Here, we show how to use the essential ingredients of the package.
=#

using Distributions
using HiddenMarkovModels
using HiddenMarkovModels: test_coherent_algorithms  #src
using LinearAlgebra
using Random
using Test  #src

#-

rng = Random.default_rng()
Random.seed!(rng, 63);

# ## Model construction

#=
The package provides a versatile `HMM` type with three attributes:
- a vector of state initialization probabilities
- a matrix of state transition probabilities
- a vector of observation distributions, one for each state

Here we keep it simple by using Distributions.jl, but there are other ways.
=#

init = [0.8, 0.2]
trans = [0.7 0.3; 0.3 0.7]
dists = [MvNormal(-ones(3), I), MvNormal(+ones(3), I)]
hmm = HMM(init, trans, dists);

# ## Simulation

#=
You can simulate a pair of state and observation sequences by specifying how long you want them to be.
=#

state_seq, obs_seq = rand(rng, hmm, 20);

#=
Note that the observation sequence is a vector, whose elements have whatever type an observation distribution returns when sampled.
As we will see, this can be more generic than just numbers or arrays.
=#

state_seq[1], obs_seq[1]

#=
In practical applications, the state sequence is not known, which is why we need inference algorithms to gather information about it.
=#

# ## Inference

#=
The Viterbi algorithm returns the most likely state sequence $\hat{X}_{1:T} = \underset{X_{1:T}}{\mathrm{argmax}}~\mathbb{P}(X_{1:T} \vert Y_{1:T})$, along with the joint loglikelihood $\mathbb{P}(\hat{X}_{1:T}, Y_{1:T})$.
=#

best_state_seq, best_joint_loglikelihood = viterbi(hmm, obs_seq);

#=
As we can see, it is very close to the true state sequence, but not necessarily equal.
=#

vcat(state_seq', best_state_seq')

#=
The forward algorithm returns a matrix of filtered state marginals $\alpha[i, t] = \mathbb{P}(X_t = i | Y_{1:t})$, along with the loglikelihood $\mathbb{P}(Y_{1:T})$ of the observation sequence.
=#

filtered_state_marginals, obs_seq_loglikelihood1 = forward(hmm, obs_seq);

#=
At each time $t$, it takes only the observations up to time $t$ into account.
This is particularly useful to infer the marginal distribution of the last state.
=#

filtered_state_marginals[:, end]

#=
Conversely, the forward-backward algorithm returns a matrix of smoothed state marginals $\gamma[i, t] = \mathbb{P}(X_t = i | Y_{1:T})$, along with the loglikelihood $\mathbb{P}(Y_{1:T})$ of the observation sequence.
=#

smoothed_state_marginals, obs_seq_loglikelihood2 = forward_backward(hmm, obs_seq);

#=
At each time $t$, it takes all observations up to time $T$ into account.
This is particularly useful during learning.
Note that forward and forward-backward only coincide at the last time step.
=#

collect(zip(filtered_state_marginals, smoothed_state_marginals))

#=
Finally, we provide a thin wrapper around the forward algorithm for observation sequence loglikelihoods $\mathbb{P}(Y_{1:T})$.
=#

logdensityof(hmm, obs_seq)

#=
The same function can also compute joint loglikelihoods $\mathbb{P}(X_{1:T}, Y_{1:T})$ that take the states into account.
=#

logdensityof(hmm, obs_seq, state_seq)

#=
For instance, we can check that the output of Viterbi is at least as likely as the true state sequence.
=#

logdensityof(hmm, obs_seq, best_state_seq)

# ## Learning

#=
The Baum-Welch algorithm is a variant of Expectation-Maximization, designed specifically to estimate HMM parameters.
Since it is a local optimization procedure, it requires a starting point that is close enough to the true model.
=#

init_guess = [0.7, 0.3]
trans_guess = [0.6 0.4; 0.4 0.6]
dists_guess = [MvNormal(-0.5 * ones(3), I), MvNormal(+0.5 * ones(3), I)]
hmm_guess = HMM(init_guess, trans_guess, dists_guess);

#=
Now we can estimate parameters based on a slightly longer sequence.
=#

_, long_obs_seq = rand(rng, hmm, 1000)
hmm_est, loglikelihood_evolution = baum_welch(hmm_guess, long_obs_seq);

#=
An essential guarantee of this algorithm is that the loglikelihood of the observation sequence keeps increasing as the model improves.
=#

first(loglikelihood_evolution), last(loglikelihood_evolution)

#=
We can check that the transition matrix has been correctly estimated.
=#

cat(transition_matrix(hmm_est), transition_matrix(hmm); dims=3)

#=
And so have the observation distributions.
=#

map(dist -> dist.μ, hcat(obs_distributions(hmm_est), obs_distributions(hmm)))

#=
On the other hand, the initialization is concentrated on one state.
This effect can be mitigated by learning from several independent sequences.
=#

hcat(initialization(hmm_est), initialization(hmm))

#=
Since HMMs are not identifiable up to a permutation of the states, there is no guarantee that state $i$ in the true model will correspond to state $i$ in the estimated model.
This is important to keep in mind when testing new models.
=#

# ## Multiple sequences

#=
In many applications, we have access to various observation sequences of different lengths.
=#

_, obs_seq2 = rand(rng, hmm, 30)
obs_seqs = [obs_seq, obs_seq2];

#=
Every algorithm in the package accepts multiple sequences in a concatenated form.
The user must also specify where each sequence ends in the concatenated vector, by passing `seq_ends` as a keyword argument.
Otherwise, the input would be treated as a single observation sequence, which is mathematically incorrect.
=#

obs_seq_concat = reduce(vcat, obs_seqs)
seq_ends = cumsum(length.(obs_seqs))

#=
The outputs of inference algorithms are then concatenated, and the associated loglikelihoods are summed over all sequences.
=#

best_state_seq_concat, _ = viterbi(hmm, obs_seq_concat; seq_ends);

#=
The function `seq_limits` returns the begin and end of a given sequence in the concatenated vector.
It can be used to untangle the results.
=#

start2, stop2 = seq_limits(seq_ends, 2)
best_state_seq_concat[start2:stop2] == first(viterbi(hmm, obs_seq2))

#=
While inference algorithms can also be run separately on each sequence without changing the results, considering multiple sequences together is nontrivial for Baum-Welch, and the package takes care of it automatically.
=#

baum_welch(hmm_guess, obs_seq_concat; seq_ends);

# ## Tests  #src

control_seq, seq_ends = fill(nothing, 1000), 100:10:1000  #src
test_coherent_algorithms(rng, hmm, hmm_guess; control_seq, seq_ends, atol=1e-1)  #src
