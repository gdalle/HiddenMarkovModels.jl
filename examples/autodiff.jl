# # Autodiff

#=
Here we show how to compute gradients of the observation sequence loglikelihood with respect to various parameters.
=#

using DensityInterface
using Distributions
using Enzyme
using ForwardDiff
using HiddenMarkovModels
using HiddenMarkovModels: test_coherent_algorithms  #src
using LinearAlgebra
using Random: Random, AbstractRNG
using StatsAPI
using Test  #src

#-

rng = Random.default_rng()
Random.seed!(rng, 63);

# ## Forward mode

#=
Since all of our code is type-generic, it is amenable to forward-mode automatic differentiation with ForwardDiff.jl.
=#

init = [0.8, 0.2]
trans = [0.7 0.3; 0.3 0.7]
means = [-1.0, 1.0]
dists = Normal.(means)
hmm = HMM(init, trans, dists);

_, obs_seq = rand(rng, hmm, 10);

#-

f1(new_init) = logdensityof(HMM(new_init, trans, dists), obs_seq)
ForwardDiff.gradient(f1, init)

#-

f2(new_trans) = logdensityof(HMM(init, new_trans, dists), obs_seq)
ForwardDiff.gradient(f2, trans)

#-

f3(new_means) = logdensityof(HMM(init, trans, Normal.(new_means)), obs_seq)
ForwardDiff.gradient(f3, means)

# ## Reverse mode

#=
In the presence of many parameters, reverse mode automatic differentiation of the loglikelihood will be much more efficient.
This requires using Enzyme.jl and the mutating `forward!` function.
=# 

# ## Gradient descent for estimation
