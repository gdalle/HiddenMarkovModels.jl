# # Autodiff

#=
Here we show how to compute gradients of the observation sequence loglikelihood with respect to various parameters.
=#

using ComponentArrays
using DensityInterface
using Distributions
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using LinearAlgebra
using Random: Random, AbstractRNG
using StableRNGs
using StatsAPI
using Test  #src
using Zygote: Zygote

#-

rng = StableRNG(63)

# ## Data generation

init = [0.8, 0.2]
trans = [0.7 0.3; 0.3 0.7]
means = [-1.0, 1.0]
dists = Normal.(means)
hmm = HMM(init, trans, dists);

#-

obs_seqs = [rand(rng, hmm, 10).obs_seq, rand(rng, hmm, 20).obs_seq];
obs_seq = reduce(vcat, obs_seqs)
seq_ends = cumsum(length.(obs_seqs));

# ## Forward mode

#=
Since all of our code is type-generic, it is amenable to forward-mode automatic differentiation with ForwardDiff.jl.

Because this backend only accepts a single vector input, we wrap all parameters with ComponentArrays.jl, and define a new function to differentiate.
=#

params = ComponentVector(; init, trans, means)

function f(params::ComponentVector)
    new_hmm = HMM(params.init, params.trans, Normal.(params.means))
    return logdensityof(new_hmm, obs_seq; seq_ends)
end;

#=
The gradient computation is now straightforward.
We will use this value as a source of truth to compare with reverse mode.
=#

grad_f = ForwardDiff.gradient(f, params)

# ## Reverse mode

#=
In the presence of many parameters, reverse mode automatic differentiation of the loglikelihood will be much more efficient.
The package includes a chain rule for `logdensityof`, which means backends like Zygote.jl can be used out of the box.
=#

grad_z = Zygote.gradient(f, params)[1]

#-

grad_f ≈ grad_z

#=
Enzyme.jl also works natively but we have to avoid the type instability of global variables by providing more information.
=#

function f_extended(params::ComponentVector, obs_seq, seq_ends)
    new_hmm = HMM(params.init, params.trans, Normal.(params.means))
    return logdensityof(new_hmm, obs_seq; seq_ends)
end;

shadow_params = Enzyme.make_zero(params)

Enzyme.autodiff(
    Enzyme.Reverse,
    f_extended,
    Enzyme.Active,
    Enzyme.Duplicated(params, shadow_params),
    Enzyme.Const(obs_seq),
    Enzyme.Const(seq_ends),
)

grad_e = shadow_params

#-

grad_e ≈ grad_f

#=
For increased efficiency, one can provide temporary storage to Enzyme.jl in order to avoid allocations.
This requires going one level deeper, by leveraging the in-place [`HiddenMarkovModels.forward!`](@ref) function.
=#

# ## Gradient methods

#=
Once we have gradients of the loglikelihood, it is a natural idea to perform gradient descent in order to fit the parameters of a custom HMM.
However, there are two caveats we must keep in mind.

First, computing a gradient essentially requires running the forward-backward algorithm, which means it is expensive.
Given the output of forward-backward, if there is a way to perform a more accurate parameter update (like going straight to the maximum likelihood value), it is probably worth it.
That is what we show in the other tutorials with the reimplementation of the `fit!` method.

Second, HMM parameters live in a constrained space, which calls for a projected gradient descent.
Most notably, the transition matrix must be stochastic, and the orthogonal projection onto this set (the Birkhoff polytope) is not easy to obtain.

Still, first order optimization can be relevant when we lack explicit formulas for maximum likelihood.
=#

# ## Tests  #src

@test grad_f ≈ grad_z  #src
@test grad_e ≈ grad_f  #src
