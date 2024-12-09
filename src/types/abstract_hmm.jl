"""
    AbstractHMM 

Abstract supertype for an HMM amenable to simulation, inference and learning.

# Interface

To create your own subtype of `AbstractHMM`, you need to implement the following methods:

- [`initialization`](@ref)
- [`transition_matrix`](@ref)
- [`obs_distributions`](@ref)
- [`fit!`](@ref) (for learning)

# Applicable functions

Any `AbstractHMM` which satisfies the interface can be given to the following functions:

- [`rand`](@ref)
- [`logdensityof`](@ref)
- [`forward`](@ref)
- [`viterbi`](@ref)
- [`forward_backward`](@ref)
- [`baum_welch`](@ref) (if `[fit!](@ref)` is implemented)
"""
abstract type AbstractHMM{ar} end

@inline DensityInterface.DensityKind(::AbstractHMM) = HasDensity()

## Interface

"""
    length(hmm)

Return the number of states of `hmm`.
"""
Base.length(hmm::AbstractHMM) = length(initialization(hmm))


"""
    eltype(hmm, obs, control)

Return a type that can accommodate forward-backward computations for `hmm` on observations similar to `obs`.

It is typically a promotion between the element type of the initialization, the element type of the transition matrix, and the type of an observation logdensity evaluated at `obs`.
"""
function Base.eltype(hmm::AbstractHMM, obs, control)
    init_type = eltype(initialization(hmm))
    trans_type = eltype(transition_matrix(hmm, control))
    dists = obs_distributions(hmm, control, obs)
    logdensity_type = typeof(logdensityof(dists[1], obs))
    return promote_type(init_type, trans_type, logdensity_type)
end

"""
    initialization(hmm)
    initialization(hmm, control)

Return the vector of initial state probabilities for `hmm` (possibly when `control` is applied).
"""
function initialization end

"""
    log_initialization(hmm)

Return the vector of initial state log-probabilities for `hmm` (possibly when `control` is applied).

Falls back on `initialization`.
"""
log_initialization(hmm::AbstractHMM) =
    elementwise_log(initialization(hmm))

"""
    transition_matrix(hmm)
    transition_matrix(hmm, control)

Return the matrix of state transition probabilities for `hmm` (possibly when `control` is applied).

!!! note
    When processing sequences, the control at time `t` influences the transition from time `t` to `t+1` (and not from time `t-1` to `t`).
"""
function transition_matrix end

"""
    log_transition_matrix(hmm)
    log_transition_matrix(hmm, control)

Return the matrix of state transition log-probabilities for `hmm` (possibly when `control` is applied).

Falls back on `transition_matrix`.

!!! note
    When processing sequences, the control at time `t` influences the transition from time `t` to `t+1` (and not from time `t-1` to `t`).
"""
log_transition_matrix(hmm::AbstractHMM, control) =
    elementwise_log(transition_matrix(hmm, control))

"""
    obs_distributions(hmm)
    obs_distributions(hmm, control)
    obs_distributions(hmm, control, obs)

Return a vector of observation distributions, one for each state of `hmm` (possibly when `control` is applied).

These distribution objects should implement

- `Random.rand(rng, dist)` for sampling
- `DensityInterface.logdensityof(dist, obs)` for inference
- `StatsAPI.fit!(dist, obs_seq, weight_seq)` for learning
"""
function obs_distributions end

## Fallbacks for no control

initialization(hmm::AbstractHMM, ::Nothing) = initialization(hmm)
transition_matrix(hmm::AbstractHMM, ::Nothing) = transition_matrix(hmm)
log_transition_matrix(hmm::AbstractHMM, ::Nothing) = log_transition_matrix(hmm)
obs_distributions(hmm::AbstractHMM, ::Nothing) = obs_distributions(hmm)
function obs_distributions(hmm::AbstractHMM, control, ::Any)
    return obs_distributions(hmm, control)
end

previous_obs(::AbstractHMM{false}, obs_seq::AbstractVector, t::Integer) = nothing
previous_obs(::AbstractHMM{true}, obs_seq::AbstractVector, t::Integer) = obs_seq[t - 1]

"""
    StatsAPI.fit!(
        hmm, fb_storage::ForwardBackwardStorage,
        obs_seq, [control_seq]; seq_ends,
    )

Update `hmm` in-place based on information generated during forward-backward.

This function is allowed to reuse `fb_storage` as a scratch space, so its contents should not be trusted afterwards.
"""
StatsAPI.fit!

## Fill logdensities

function obs_logdensities!(
    logb::AbstractVector{T}, hmm::AbstractHMM, obs, control, prev_obs
) where {T}
    dists = obs_distributions(hmm, control, prev_obs)
    @simd for i in eachindex(logb, dists)
        logb[i] = logdensityof(dists[i], obs)
    end
    @argcheck maximum(logb) < typemax(T)
    return nothing
end

## Sampling

"""
    rand([rng,] hmm, T)
    rand([rng,] hmm, control_seq)

Simulate `hmm` for `T` time steps, or when the sequence `control_seq` is applied.
    
Return a named tuple `(; state_seq, obs_seq)`.
"""
function Random.rand(rng::AbstractRNG, hmm::AbstractHMM, control_seq::AbstractVector)
    T = length(control_seq)
    dummy_log_probas = fill(-Inf, length(hmm))

    init = initialization(hmm)
    state_seq = Vector{Int}(undef, T)
    state1 = rand(rng, LightCategorical(init, dummy_log_probas))
    state_seq[1] = state1

    @views for t in 1:(T - 1)
        trans = transition_matrix(hmm, control_seq[t])
        state_seq[t + 1] = rand(
            rng, LightCategorical(trans[state_seq[t], :], dummy_log_probas)
        )
    end

    dists1 = obs_distributions(hmm, control_seq[1], missing)
    obs1 = rand(rng, dists1[state1])
    obs_seq = Vector{typeof(obs1)}(undef, T)
    obs_seq[1] = obs1

    for t in 2:T
        dists = obs_distributions(hmm, control_seq[t], previous_obs(hmm, obs_seq, t))
        obs_seq[t] = rand(rng, dists[state_seq[t]])
    end
    return (; state_seq=state_seq, obs_seq=obs_seq)
end

function Random.rand(hmm::AbstractHMM, control_seq::AbstractVector)
    return rand(default_rng(), hmm, control_seq)
end

function Random.rand(rng::AbstractRNG, hmm::AbstractHMM, T::Integer)
    return rand(rng, hmm, Fill(nothing, T))
end

function Random.rand(hmm::AbstractHMM, T::Integer)
    return rand(hmm, Fill(nothing, T))
end

## Prior

"""
    logdensityof(hmm)

Return the prior loglikelihood associated with the parameters of `hmm`.
"""
DensityInterface.logdensityof(hmm::AbstractHMM) = false
