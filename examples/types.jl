# # Types

#=
Here we explain why playing with different number and array types can be useful in an HMM.
=#

using Distributions
using HiddenMarkovModels
using HiddenMarkovModels: log_transition_matrix  #src
using HMMTest  #src
using LinearAlgebra
using LogarithmicNumbers
using Measurements
using Random
using SparseArrays
using StableRNGs
using Test  #src

#-

rng = StableRNG(63);

# ## General principle

#=
The whole package is agnostic with respect to types, it performs the right promotions automatically.
Therefore, the types we get in the output only depend only on the types present in the input HMM and the observation sequences.
=#

# ## Weird number types

#=
A wide variety of number types can be plugged into HMM parameters to enhance precision or change inference behavior.
Some examples are:
- `BigFloat` for arbitrary precision
- [LogarithmicNumbers.jl](https://github.com/cjdoris/LogarithmicNumbers.jl) to increase numerical stability
- [Measurements.jl](https://github.com/JuliaPhysics/Measurements.jl) to propagate uncertainties
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) for dual numbers in automatic differentiation

To give an example, let us first generate some data from a vanilla HMM.
=#

init = [0.6, 0.4]
trans = [0.7 0.3; 0.2 0.8]
dists = [Normal(-1.0), Normal(1.0)]
hmm = HMM(init, trans, dists)
state_seq, obs_seq = rand(rng, hmm, 100);

#=
Now we construct a new HMM with some uncertainty on the observation means, using Measurements.jl.
Note that uncertainty on the transition parameters would throw an error because the matrix has to be stochastic.
=#

dists_guess = [Normal(-1.0 ± 0.1), Normal(1.0 ± 0.2)]
hmm_uncertain = HMM(init, trans, dists_guess)

#=
Every quantity we compute with this new HMM will have propagated uncertainties around it.
=#

logdensityof(hmm, obs_seq)

#-

logdensityof(hmm_uncertain, obs_seq)

#=
We can check that the interval is centered around the true value.
=#

Measurements.value(logdensityof(hmm_uncertain, obs_seq)) ≈ logdensityof(hmm, obs_seq)

#=
!!! warning "Number types in Baum-Welch"
    For now, the Baum-Welch algorithm will generally fail with custom number types due to promotion.
    The reason is that if some parameters have type `T1` and some `T2`, the forward-backward algorithm will compute quantities of type `T = promote_type(T1, T2)`.
    These quantities may not be suited to the existing containers inside an HMM, and since updates happen in-place for performance, we cannot create a new one.
    Suggestions are welcome to fix this issue.
=#

# ## Tests #src

@test Measurements.value(logdensityof(hmm_uncertain, obs_seq)) ≈ logdensityof(hmm, obs_seq)  #src

# ## Sparse matrices

#=
[Sparse matrices](https://docs.julialang.org/en/v1/stdlib/SparseArrays/) are very useful for large models, because it means the memory and computational requirements will scale as the number of possible transitions.
In general, this number is much smaller than the square of the number of states.
=#

#=
We can easily construct an HMM with a sparse transition matrix, where some transitions are structurally forbidden.
=#

trans = sparse([
    0.7 0.3 0
    0 0.7 0.3
    0.3 0 0.7
])

#-

init = [0.2, 0.6, 0.2]
dists = [Normal(1.0), Normal(2.0), Normal(3.0)]
hmm = HMM(init, trans, dists)

#=
When we simulate it, the transitions outside of the nonzero coefficients simply cannot happen.
=#

state_seq, obs_seq = rand(rng, hmm, 1000)
state_transitions = collect(zip(state_seq[1:(end - 1)], state_seq[2:end]));

#=
For a possible transition:
=#

count(isequal((2, 2)), state_transitions)

#=
For an impossible transition:
=#

count(isequal((2, 1)), state_transitions)

#=
Now we apply Baum-Welch from a guess with the right sparsity pattern.
=#

init_guess = [0.3, 0.4, 0.3]
trans_guess = sparse([
    0.6 0.4 0
    0 0.6 0.4
    0.4 0 0.6
])
dists_guess = [Normal(1.1), Normal(2.1), Normal(3.1)]
hmm_guess = HMM(init_guess, trans_guess, dists_guess);

#-

hmm_est, loglikelihood_evolution = baum_welch(hmm_guess, obs_seq);
first(loglikelihood_evolution), last(loglikelihood_evolution)

#=
The estimated model has kept the same sparsity pattern as the guess.
=#

transition_matrix(hmm_est)

#=
Another useful array type is [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl), which reduces allocations for small state spaces.
=#

# ## Tests  #src

@test nnz(log_transition_matrix(hmm)) == nnz(transition_matrix(hmm))  #src

seq_ends = cumsum(rand(rng, 100:200, 100));  #src
control_seq = fill(nothing, last(seq_ends));  #src
test_identical_hmmbase(rng, hmm, 100; hmm_guess)  #src
test_coherent_algorithms(rng, hmm, control_seq; seq_ends, hmm_guess, init=false, atol=0.08)  #src
test_type_stability(rng, hmm, control_seq; seq_ends, hmm_guess)  #src
# https://github.com/JuliaSparse/SparseArrays.jl/issues/469  #src
@test_skip test_allocations(rng, hmm, control_seq; seq_ends, hmm_guess)  #src
