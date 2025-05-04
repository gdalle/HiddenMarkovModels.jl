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
#md using CairoMakie
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

function HMMs.transition_matrix(hmm::ARGaussianHMM, _prev_obs::Union{Real,Missing})
    return hmm.trans
end

function HMMs.obs_distributions(hmm::ARGaussianHMM, prev_obs::Union{Real,Missing})
    means_by_state = [hmm.a[i] * coalesce(prev_obs, 0.0) + hmm.b[i] for i in 1:length(hmm)]
    return Normal.(means_by_state, 0.2)
end

#=
In this case, the transition matrix does not depend on the previous observation.
=#

# ## Simulation

init = [0.6, 0.4]
trans = [0.95 0.05; 0.05 0.95]
a = [0.7, 0.8]
b = [+1.0, -1.0]
hmm = ARGaussianHMM(init, trans, a, b);

#=
Simulation requires a manual procedure which reinjects the last observation as a control variable.
=#

function simulate_autoregressive(rng::AbstractRNG, hmm::AbstractHMM, T::Integer)
    init = initialization(hmm)
    first_dists = obs_distributions(hmm, missing)
    first_state = rand(rng, Distributions.Categorical(init))
    first_obs = rand(rng, first_dists[first_state])
    state_seq = [first_state]
    obs_seq = [first_obs]

    for t in 2:T
        prev_state = state_seq[t - 1]
        prev_obs = obs_seq[t - 1]
        trans = transition_matrix(hmm, prev_obs)
        dists = obs_distributions(hmm, prev_obs)
        new_state = rand(rng, Distributions.Categorical(trans[prev_state, :]))
        new_obs = rand(rng, dists[new_state])
        push!(state_seq, new_state)
        push!(obs_seq, new_obs)
    end

    return (; state_seq=state_seq, obs_seq=obs_seq)
end

#-

T = 300
times = 1:T
state_seq, obs_seq = simulate_autoregressive(rng, hmm, T)

#md let
#md     fig = Figure()
#md     ax = Axis(fig[1, 1]; xlabel="time", ylabel="observation")
#md     scatter!(ax, times[state_seq .== 1], obs_seq[state_seq .== 1]; label="state 1")
#md     scatter!(ax, times[state_seq .== 2], obs_seq[state_seq .== 2]; label="state 2")
#md     axislegend(ax)
#md     fig
#md end

@test mean(y -> isapprox(y, b[1] / (1 - a[1]); rtol=0.1), obs_seq[state_seq .== 1]) > 0.3  #src
@test mean(y -> isapprox(y, b[2] / (1 - a[2]); rtol=0.1), obs_seq[state_seq .== 2]) > 0.3  #src

# ## Inference

#=
At inference time, the observations are considered fixed.
Therefore, we are allowed to use them as controls.
=#

control_seq = vcat(missing, obs_seq[1:(end - 1)])

#=
We show an example with Viterbi's algorithm.
=#

best_state_seq, _ = viterbi(hmm, obs_seq, control_seq)

#md let
#md     fig = Figure()
#md     ax0 = Axis(fig[0, 1]; ylabel="observations")
#md     ax1 = Axis(fig[1, 1]; limits=(nothing, (0.5, 2.5)), yticks=1:2, ylabel="true state")
#md     ax2 = Axis(
#md         fig[2, 1];
#md         limits=(nothing, (0.5, 2.5)),
#md         yticks=1:2,
#md         ylabel="inferred state",
#md         xlabel="time",
#md     )
#md     scatter!(ax0, times, obs_seq)
#md     lines!(ax1, times, state_seq)
#md     lines!(ax2, times, best_state_seq)
#md     fig
#md end

@test mean(best_state_seq .== state_seq) > 0.95  #src
