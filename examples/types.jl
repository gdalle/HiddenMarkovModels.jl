# # Types

#=
Here we explain why playing with different number and array types can be useful in an HMM.
=#

using Distributions
using HiddenMarkovModels
using HMMTest  #src
using LinearAlgebra
using LogarithmicNumbers
using Random
using SparseArrays
using Test  #src

#-

rng = Random.default_rng()
Random.seed!(rng, 63);

# ## Logarithmic numbers

#=
!!! warning
    Work in progress
=#

# ## Sparse arrays

#=
Using sparse matrices is very useful for large models, because it means the memory and computational requirements will scale as the number of possible transitions.
In general, this number is much smaller than the square of the number of states.
=#

#=
We can easily construct an HMM with a sparse transition matrix, where some transitions are structurally forbidden.
=#

init = [0.2, 0.6, 0.2]
trans = sparse([
    0.8 0.2 0.0
    0.0 0.8 0.2
    0.2 0.0 0.8
])
dists = [Normal(-2.0), Normal(0.0), Normal(+2.0)]
hmm = HMM(init, trans, dists);

#=
When we simulate it, the transitions outside of the nonzero coefficients simply cannot happen.
=#

state_seq, obs_seq = rand(rng, hmm, 1000)
state_transitions = collect(zip(state_seq[1:(end - 1)], state_seq[2:end]));

#-

count(isequal((2, 2)), state_transitions)

#-

count(isequal((2, 1)), state_transitions)

#=
Now we apply Baum-Welch from a guess with the right sparsity pattern.
=#

init_guess = [0.3, 0.4, 0.3]
trans_guess = sparse([
    0.7 0.3 0.0
    0.0 0.7 0.3
    0.3 0.0 0.7
])
dists_guess = [Normal(-1.5), Normal(0.0), Normal(+1.5)]
hmm_guess = HMM(init_guess, trans_guess, dists_guess);

#-

hmm_est, loglikelihood_evolution = baum_welch(hmm_guess, obs_seq);
first(loglikelihood_evolution), last(loglikelihood_evolution)

#=
The estimated model has kept the same sparsity pattern.
=#

transition_matrix(hmm_est)

#-

transition_matrix(hmm)

# ## Tests  #src

control_seqs = [fill(nothing, rand(rng, 100:200)) for k in 1:100];  #src
control_seq = reduce(vcat, control_seqs);  #src
seq_ends = cumsum(length.(control_seqs));  #src

test_identical_hmmbase(rng, hmm, hmm_guess; T=100)  #src
test_coherent_algorithms(rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.05, init=false)  #src
test_type_stability(rng, hmm, hmm_guess; control_seq, seq_ends)  #src
# https://github.com/JuliaSparse/SparseArrays.jl/issues/469  #src
@test_skip test_allocations(rng, hmm, hmm_guess; control_seq, seq_ends)  #src
