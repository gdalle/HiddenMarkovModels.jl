"""
    AbstractHMM 

Abstract supertype for an HMM amenable to simulation, inference and learning.

# Interface

To create your own subtype of `AbstractHiddenMarkovModel`, you need to implement the following methods:

- [`length(hmm)`](@ref)
- [`eltype(hmm, obs)`](@ref)
- [`initialization(hmm)`](@ref)
- [`transition_matrix(hmm, t)`](@ref)
- [`obs_distributions(hmm, t)`](@ref)

# Applicable functions

Any HMM object which satisfies the interface can be given as input to the following functions:

- [`rand(rng, hmm, T)`](@ref)
- [`logdensityof(hmm, obs_seq)`](@ref)
- [`forward(hmm, obs_seq)`](@ref)
- [`viterbi(hmm, obs_seq)`](@ref)
- [`forward_backward(hmm, obs_seq)`](@ref)

# Fitting
"""
abstract type AbstractHMM end

@inline DensityInterface.DensityKind(::AbstractHMM) = HasDensity()

## Interface

"""
    length(hmm)

Return the number of states of `hmm`.
"""
Base.length

"""
    eltype(hmm, obs)

Return a type that can accommodate forward-backward computations for `hmm` on observations similar to `obs`.

It is typically a promotion between the element type of the initialization, the element type of the transition matrix, and the type of an observation logdensity evaluated at `obs`.
"""
function Base.eltype(hmm::AbstractHMM, obs)
    init_type = eltype(initialization(hmm))
    trans_type = eltype(transition_matrix(hmm, 1))
    logdensity_type = typeof(logdensityof(obs_distributions(hmm, 1)[1], obs))
    return promote_type(init_type, trans_type, logdensity_type)
end

"""
    initialization(hmm)

Return the vector of initial state probabilities for `hmm`.
"""
function initialization end

"""
    transition_matrix(hmm, t)

Return the matrix of state transition probabilities for `hmm` at time `t`.
"""
function transition_matrix end

"""
    obs_distributions(hmm, t)

Return a vector of observation distributions, one for each state of `hmm` at time `t`.

There objects should support `rand(rng, dist)` and `DensityInterface.logdensityof(dist, obs)`.
"""
function obs_distributions end

function obs_logdensities!(logb::AbstractVector, hmm::AbstractHMM, t::Integer, obs)
    dists = obs_distributions(hmm, t)
    @inbounds for i in eachindex(logb, dists)
        logb[i] = logdensityof(dists[i], obs)
    end
    return check_right_finite(logb)
end

"""
    fit!(hmm, ...)

Update `hmm` in-place based on information generated during forward-backward.
"""
StatsAPI.fit!  # TODO: complete

## Sampling

"""
    rand([rng,] hmm, T)

Simulate `hmm` for `T` time steps. 
"""
function Base.rand(rng::AbstractRNG, hmm::AbstractHMM, T::Integer)
    dummy_log_probas = fill(-Inf, length(hmm))

    init = initialization(hmm)
    state1 = rand(rng, LightCategorical(init, dummy_log_probas))
    state_seq = Vector{typeof(state1)}(undef, T)
    state_seq[1] = state1

    dists = obs_distributions(hmm, 1)
    obs1 = rand(rng, dists[state1])
    obs_seq = Vector{typeof(obs1)}(undef, T)
    obs_seq[1] = obs1

    @views for t in 2:T
        trans = transition_matrix(hmm, t)
        dists = obs_distributions(hmm, t)
        state_seq[t] = rand(
            rng, LightCategorical(trans[state_seq[t - 1], :], dummy_log_probas)
        )
        obs_seq[t] = rand(rng, dists[state_seq[t]])
    end
    return (; state_seq=state_seq, obs_seq=obs_seq)
end

Base.rand(hmm::AbstractHMM, T::Integer) = rand(default_rng(), hmm, T)
