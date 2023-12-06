# # Distributions

using DensityInterface
using Distributions
using HiddenMarkovModels
using Random: Random, AbstractRNG
using StatsAPI
using Test  #src

#-

mutable struct PoissonProcess{R}
    λ::R
end

DensityInterface.DensityKind(::PoissonProcess) = HasDensity()

function Random.rand(rng::AbstractRNG, pp::PoissonProcess)
    nb_events = rand(rng, Poisson(pp.λ))
    event_times = rand(rng, Uniform(0, 1), nb_events)
    return event_times
end

function DensityInterface.logdensityof(pp::PoissonProcess, event_times::Vector)
    return -pp.λ + length(event_times) * log(pp.λ)
end

function StatsAPI.fit!(pp::PoissonProcess, x, w)
    pp.λ = sum(length(xᵢ) * wᵢ for (xᵢ, wᵢ) in zip(x, w)) / sum(w)
    return nothing
end

#-

init = [0.3, 0.7]
trans = [0.8 0.2; 0.1 0.9]
dists = [PoissonProcess(1.0), PoissonProcess(5.0)]

hmm = HMM(init, trans, dists)

T = 100
state_seq, obs_seq = rand(hmm, T)

#-

forward_backward(hmm, obs_seq)

#-

init_guess = [0.5, 0.5]
trans_guess = [0.5 0.5; 0.5 0.5]
dists_guess = [PoissonProcess(2.0), PoissonProcess(3.0)]
hmm_guess = HMM(init_guess, trans_guess, dists_guess)

hmm_est, logL_evolution = baum_welch(hmm_guess, obs_seq)

@test hmm_est.dists[1].λ ≈ hmm.dists[1].λ atol = 0.5  #src
@test hmm_est.dists[2].λ ≈ hmm.dists[2].λ atol = 0.5  #src
