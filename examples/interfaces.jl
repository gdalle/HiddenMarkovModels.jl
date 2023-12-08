# # Interfaces

#=
Here we discuss how to extend the observation distributions or HMM behavior to fit specific needs.
=#

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

# ## Creating a new distribution

#=
In an `HMM` object, the observation distributions do not need to come from Distributions.jl.
They only need to implement three methods:
- `Random.rand(rng, dist)` for sampling
- `DensityInterface.logdensityof(dist, obs)` for inference
- `StatsAPI.fit!(dist, obs_seq, weight_seq)` for learning

In addition, the observation can be arbitrary Julia types.
So let's construct a distribution that generates stuff.

If you want more sophisticated examples, check out the definitions of `HiddenMarkovModels.LightDiagNormal` and `HiddenMarkovModels.LightCategorical`, which are designed to be fast and allocation-free.
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
It is important to declare to DensityInterface.jl that the custom distribution has a density, thanks to the following trait.
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

#=
Let's put it to the test.
=#

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

control_seq, seq_ends = fill(nothing, 1000), 100:10:1000  #src
test_coherent_algorithms(rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.2, init=false)  #src

# ## Creating a new HMM type

#=
In some scenarios, the vanilla Baum-Welch algorithm is not exactly what we want.
For instance, we might have a prior on the parameters of our model, which we want to apply during the fitting step of the iterative procedure.

Then we need to create a new type that satisfies the `AbstractHMM` interface.
Let's make a simpler version of the built-in `HMM`m with a prior saying that each transition has been observed exactly once.
=#

struct PriorHMM{T,D} <: AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dists::Vector{D}
end

#=
The basic requirements for `AbstractHMM` are the following three functions.
=#

HiddenMarkovModels.initialization(hmm::PriorHMM) = hmm.init
HiddenMarkovModels.transition_matrix(hmm::PriorHMM) = hmm.trans
HiddenMarkovModels.obs_distributions(hmm::PriorHMM) = hmm.dists

#=
In addition, we want to overload `logdensityof` to specify our prior loglikelihood.
=#

function DensityInterface.logdensityof(hmm::PriorHMM)
    prior = Dirichlet(ones(length(hmm)))
    return sum(logdensityof(prior, row) for row in eachrow(transition_matrix(hmm)))
end

#=
And finally, we redefine the specific method of `fit!` that is used during Baum-Welch.
It accepts the same inputs as `baum_welch` for multiple sequences (disregard `control_seq` for now), and an additional `fb_storage` containing the results of the forward-backward algorithm.

The goal is to modify `hmm` in-place to update its parameters with their current maximum likelihood estimates.
We will make use of the attributes `fb_storage.γ` and `fb_storage.ξ`, which contain the state and transition marginals `γ[i, t]` and `ξ[t][i, j]` at each time step.
=#

function StatsAPI.fit!(
    hmm::PriorHMM,
    obs_seq::AbstractVector;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    fb_storage::HiddenMarkovModels.ForwardBackwardStorage,
)
    hmm.init .= 0
    hmm.trans .= 1  # this is where the prior comes in
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        hmm.init .+= fb_storage.γ[:, t1]
        hmm.trans .+= sum(fb_storage.ξ[t1:t2])
    end
    hmm.init ./= sum(hmm.init)
    hmm.trans ./= sum(hmm.trans; dims=2)

    for i in 1:length(hmm)
        weight_seq = storage.γ[:, i]
        fit!(hmm.dists[i], obs_seq, weight_seq)
    end
    return nothing
end

#=
Some distributions, such as those from Distributions.jl
- do not support in-place fitting
- might expect different formats, e.g. higher-order arrays instead of a vector of objects

The function `HiddenMarkovModels.fit_in_sequence!` is a replacement for `fit!` which you can overload at will.
It is already designed to handle Distributions.jl.
=#
