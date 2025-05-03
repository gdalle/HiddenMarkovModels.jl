# # Autoregression

#=
Here, we give a example of autoregressive HMM, where the observation can depend on the previous observation.
We achieve this by abusing the control mechanism, and slightly tweaking the simulation procedure.
=#

using Distributions
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using HMMTest  #src
using LinearAlgebra
using Random
using StableRNGs
using StatsAPI
using Test  #src

#-

rng = StableRNG(63);

# ## Model

#=
We define a new subtype of `AbstractHMM` (see [Custom HMM structures](@ref)), which has state-dependent coefficients linking the previous observation to the next observation.
=#

struct ARGaussianHMM{T} <: AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    a::Vector{T}
    b::Vector{T}
end

#=
In state $i$, the observation is given by the linear model $y_t \sim \mathcal{N}(a_i y_{t-1} + b_i, 1)$.
At the first time step, there will be no previous observation $y_{t-1}$, so we also allow the value `missing`.
=#

function HMMs.initialization(hmm::ARGaussianHMM)
    return hmm.init
end

function HMMs.transition_matrix(hmm::ARGaussianHMM, _obs_prev::Union{Real,Missing})
    return hmm.trans
end

function HMMs.obs_distributions(hmm::ARGaussianHMM, obs_prev::Union{Real,Missing})
    return [
        Normal(hmm.a[i] * coalesce(obs_prev, 0.0) + hmm.b[i], 1.0) for i in 1:length(hmm)
    ]
end

#=
In this case, the transition matrix does not depend on the previous observation.
=#

# ## Simulation

d = 3
init = [0.6, 0.4]
trans = [0.7 0.3; 0.2 0.8]
a = [0.9, -0.8]
b = [-0.1, 0.2]
hmm = ARGaussianHMM(init, trans, a, b);

#=
Simulation requires a manual procedure which reinjects the last observation as a control variable.
=#

function simulate_autoregressive(rng::AbstractRNG, hmm::AbstractHMM, T::Integer)
    obs_prev = missing
    init = initialization(hmm)
    dists1 = obs_distributions(hmm, obs_prev)
    state1 = rand(rng, Categorical(init))
    obs1 = rand(rng, dists1[state1])
    obs_seq = [obs1]
    state_seq = [state1]

    for t in 1:(T - 1)
        obs_prev = obs_seq[t]  # not right
        trans = transition_matrix(hmm, obs_prev)
        push!(state_seq, rand(rng, Categorical(trans[state_seq[t], :])))
        dists = obs_distributions(hmm, obs_prev)
        push!(obs_seq, rand(rng, dists[state_seq[t + 1]]))
    end

    return (; state_seq=state_seq, obs_seq=obs_seq)
end

#-

obs_seq = simulate_autoregressive(rng, hmm, 100)
