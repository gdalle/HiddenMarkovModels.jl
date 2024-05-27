# # Basics

#=
Here we show how to use the essential ingredients of the package.
=#

using Distributions
using HiddenMarkovModels
using HMMTest  #src
using LinearAlgebra
using Random
using StableRNGs
using Test  #src

#-

rng = StableRNG(63);

# ## Model

#=
The package provides a versatile [`HMM`](@ref) type with three main attributes:
- a vector `init` of state initialization probabilities
- a matrix `trans` of state transition probabilities
- a vector `dists` of observation distributions, one for each state

Any scalar- or vector-valued distribution from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) can be used for the last part, as well as [Custom distributions](@ref).
=#

init = [0.6, 0.4]
trans = [0.7 0.3; 0.2 0.8]
dists = [MvNormal([-0.5, -0.8], I), MvNormal([0.5, 0.8], I)]
hmm = HMM(init, trans, dists)

# ## Simulation

#=
You can simulate a pair of state and observation sequences with [`rand`](@ref) by specifying how long you want them to be.
=#

T = 20
state_seq, obs_seq = rand(rng, hmm, T);

#=
The state sequence is a vector of integers.
=#

state_seq[1:3]

#=
The observation sequence is a vector whose elements have whatever type an observation distribution returns when sampled.
Here we chose a multivariate normal distribution, so we get vectors at each time step.

!!! warning "Difference from HMMBase.jl"
    In the case of multivariate observations, HMMBase.jl works with matrices, whereas HiddenMarkovModels.jl works with vectors of vectors.
    This allows us to accept more generic observations than just numbers or vectors inside the sequence.
=#

obs_seq[1:3]

#=
In practical applications, the state sequence is not known, which is why we need inference algorithms to gather information about it.
=#

# ## Inference

#=
The **Viterbi algorithm** ([`viterbi`](@ref)) returns:
- the most likely state sequence $\hat{X}_{1:T} = \underset{X_{1:T}}{\mathrm{argmax}}~\mathbb{P}(X_{1:T} \vert Y_{1:T})$,
- the joint loglikelihood $\mathbb{P}(\hat{X}_{1:T}, Y_{1:T})$ (in a vector of size 1).
=#

best_state_seq, best_joint_loglikelihood = viterbi(hmm, obs_seq);
only(best_joint_loglikelihood)

#=
As we can see, the most likely state sequence is very close to the true state sequence, but not necessarily equal.
=#

(state_seq .== best_state_seq)'

#=
The **forward algorithm** ([`forward`](@ref)) returns:
- a matrix of filtered state marginals $\alpha[i, t] = \mathbb{P}(X_t = i | Y_{1:t})$,
- the loglikelihood $\mathbb{P}(Y_{1:T})$ of the observation sequence (in a vector of size 1).
=#

filtered_state_marginals, obs_seq_loglikelihood_f = forward(hmm, obs_seq);
only(obs_seq_loglikelihood_f)

#=
At each time $t$, these filtered marginals take only the observations up to time $t$ into account.
This is particularly useful to infer the marginal distribution of the last state.
=#

filtered_state_marginals[:, T]

#=
The forward-backward algorithm ([`forward_backward`](@ref)) returns:
- a matrix of smoothed state marginals $\gamma[i, t] = \mathbb{P}(X_t = i | Y_{1:T})$,
- the loglikelihood $\mathbb{P}(Y_{1:T})$ of the observation sequence (in a vector of size 1).
=#

smoothed_state_marginals, obs_seq_loglikelihood_fb = forward_backward(hmm, obs_seq);
only(obs_seq_loglikelihood_fb)

#=
At each time $t$, it takes all observations up to time $T$ into account.
This is particularly useful during learning.
Note that forward and forward-backward only coincide at the last time step.
=#

filtered_state_marginals[:, T - 1] ≈ smoothed_state_marginals[:, T - 1]

#-

filtered_state_marginals[:, T] ≈ smoothed_state_marginals[:, T]

#=
Finally, we provide a thin wrapper ([`logdensityof`](@ref)) around the forward algorithm for observation sequence loglikelihoods $\mathbb{P}(Y_{1:T})$.
=#

logdensityof(hmm, obs_seq)

#=
Another function ([`joint_logdensityof`](@ref)) can compute joint loglikelihoods $\mathbb{P}(X_{1:T}, Y_{1:T})$ which take the states into account.
=#

joint_logdensityof(hmm, obs_seq, state_seq)

#=
For instance, we can check that the output of Viterbi is at least as likely as the true state sequence.
=#

joint_logdensityof(hmm, obs_seq, best_state_seq)

# ## Learning

#=
The Baum-Welch algorithm ([`baum_welch`](@ref)) is a variant of Expectation-Maximization, designed specifically to estimate HMM parameters.
Since it is a local optimization procedure, it requires a starting point that is close enough to the true model.
=#

init_guess = [0.5, 0.5]
trans_guess = [0.6 0.4; 0.3 0.7]
dists_guess = [MvNormal([-0.4, -0.7], I), MvNormal([0.4, 0.7], I)]
hmm_guess = HMM(init_guess, trans_guess, dists_guess);

#=
Let's estimate parameters based on a slightly longer sequence.
=#

_, long_obs_seq = rand(rng, hmm, 200)
hmm_est, loglikelihood_evolution = baum_welch(hmm_guess, long_obs_seq);

#=
An essential guarantee of this algorithm is that the loglikelihood of the observation sequence keeps increasing as the model improves.
=#

first(loglikelihood_evolution), last(loglikelihood_evolution)

#=
We can check that the transition matrix estimate has improved.
=#

cat(transition_matrix(hmm_est), transition_matrix(hmm); dims=3)

#=
And so have the estimates for the observation distributions.
=#

map(mean, hcat(obs_distributions(hmm_est), obs_distributions(hmm)))

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

nb_seqs = 100
long_obs_seqs = [last(rand(rng, hmm, rand(rng, 100:200))) for k in 1:nb_seqs];
typeof(long_obs_seqs)

#=
Every algorithm in the package accepts multiple sequences in a concatenated form.
The user must also specify where each sequence ends in the concatenated vector, by passing `seq_ends` as a keyword argument.
Otherwise, the input will be treated as a unique observation sequence, which is mathematically incorrect.
=#

long_obs_seq_concat = reduce(vcat, long_obs_seqs)
typeof(long_obs_seq_concat)

#-

seq_ends = cumsum(length.(long_obs_seqs))
seq_ends'

#=
The outputs of inference algorithms are then concatenated, and the associated loglikelihoods are split by sequence (in a vector of size `length(seq_ends)`).
=#

best_state_seq_concat, best_joint_loglikelihood_concat = viterbi(
    hmm, long_obs_seq_concat; seq_ends
);

#-

length(best_joint_loglikelihood_concat) == length(seq_ends)

#-

length(best_state_seq_concat) == last(seq_ends)

#=
The function [`seq_limits`](@ref) returns the begin and end of a given sequence in the concatenated vector.
It can be used to untangle the results.
=#

start2, stop2 = seq_limits(seq_ends, 2)

#-

best_state_seq_concat[start2:stop2] == first(viterbi(hmm, long_obs_seqs[2]))

#=
While inference algorithms can also be run separately on each sequence without changing the results, considering multiple sequences together is nontrivial for Baum-Welch.
That is why the package takes care of it automatically.
=#

hmm_est_concat, _ = baum_welch(hmm_guess, long_obs_seq_concat; seq_ends);

#=
Our estimate should be a little better.
=#

cat(transition_matrix(hmm_est_concat), transition_matrix(hmm); dims=3)

#-

map(mean, hcat(obs_distributions(hmm_est_concat), obs_distributions(hmm)))

#-

hcat(initialization(hmm_est_concat), initialization(hmm))

# ## Tests  #src

control_seq = fill(nothing, last(seq_ends));  #src
test_identical_hmmbase(rng, hmm, 100; hmm_guess)  #src
test_identical_hmmbase(rng, transpose_hmm(hmm), 100; transpose_hmm(hmm_guess))  #src
test_coherent_algorithms(rng, hmm, control_seq; seq_ends, hmm_guess)  #src
test_type_stability(rng, hmm, control_seq; seq_ends, hmm_guess)  #src
