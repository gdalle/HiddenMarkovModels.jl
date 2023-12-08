# # Distributions

using DensityInterface
using HiddenMarkovModels
using HiddenMarkovModels: test_coherent_algorithms  #src
using LinearAlgebra
using Random: Random, AbstractRNG
using StatsAPI
using Test  #src

#-

rng = Random.default_rng()
Random.seed!(rng, 63);

# ## Making your own distribution

#=
In an `HMM` object, the observation distributions do not need to come from Distributions.jl.
They only need to implement three methods:
- `Random.rand(rng, dist)` for sampling
- `DensityInterface.logdensityof(dist, obs)` for inference
- `StatsAPI.fit!(dist, obs_seq, weight_seq)` for learning

In addition, the observation can be arbitrary Julia types.
So let's construct a distribution that generates stuff.
=#

struct Stuff{T}
    quantity::T
end

#=
The distribution will only be a wrapper for a normal distribution on the quantity.
=#

mutable struct StuffDist{T}
    quantity_mean::T
end

#=
Simulation is fairly easy.
=#

function Random.rand(rng::AbstractRNG, dist::StuffDist)
    quantity = dist.quantity_mean + randn(rng)
    return Stuff(quantity)
end

#=
It is important to declare to DensityInterface.jl that the custom distribution has a density, with the following trait.
=#

DensityInterface.DensityKind(::StuffDist) = HasDensity()

#=
The logdensity itself can be computed up to an additive constant without issue.
=#

function DensityInterface.logdensityof(dist::StuffDist, obs::Stuff)
    return -abs2(obs.quantity - dist.quantity_mean)
end

#=
Finally, the fitting procedure must happen in place, and take a sequence of weighted samples.
=#

function StatsAPI.fit!(
    dist::StuffDist, obs_seq::AbstractVector{<:Stuff}, weight_seq::AbstractVector{<:Real}
)
    dist.quantity_mean =
        sum(weight * obs.quantity for (obs, weight) in zip(obs_seq, weight_seq)) /
        sum(weight_seq)
    return nothing
end

# ## Using your own distribution

init = [0.8, 0.2]
trans = [0.7 0.3; 0.3 0.7]
dists = [StuffDist(-1.0), StuffDist(+1.0)]
hmm = HMM(init, trans, dists);

#=
When we sample an observation sequence, we get a vector of `Stuff`.
=#

state_seq, obs_seq = rand(rng, hmm, 100)
eltype(obs_seq)

#=
And we can pass these observations to all of our inference algorithms.
=#

viterbi(hmm, obs_seq)

#=
If we implement `fit!`, Baum-Welch also works seamlessly.
=#

init_guess = [0.7, 0.3]
trans_guess = [0.6 0.4; 0.4 0.6]
dists_guess = [StuffDist(-0.5), StuffDist(+0.5)]
hmm_guess = HMM(init_guess, trans_guess, dists_guess);

hmm_est, loglikelihood_evolution = baum_welch(hmm_guess, obs_seq)
obs_distributions(hmm_est)

# ## Tests  #src

control_seq, seq_ends = fill(nothing, 1000), 10:10:1000  #src
test_coherent_algorithms(rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.2)  #src
