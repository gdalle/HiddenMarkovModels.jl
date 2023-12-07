"""
    AbstractHMM 

Abstract supertype for an HMM amenable to simulation, inference and learning.

# Interface

To create your own subtype of `AbstractHiddenMarkovModel`, you need to implement the following methods:

- [`initialization(hmm)`](@ref)
- [`transition_matrix(hmm, control)`](@ref)
- [`obs_distributions(hmm, control)`](@ref)

# Applicable functions

Any HMM object which satisfies the interface can be given as input to the following functions:

- [`rand(rng, hmm, control_seq)`](@ref)
- [`logdensityof(hmm, obs_seq, control_seq)`](@ref)
- [`forward(hmm, obs_seq, control_seq)`](@ref)
- [`viterbi(hmm, obs_seq, control_seq)`](@ref)
- [`forward_backward(hmm, obs_seq, control_seq)`](@ref)
"""
abstract type AbstractHMM end

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
    dist = obs_distributions(hmm, control)[1]
    logdensity_type = typeof(logdensityof(dist, obs))
    return promote_type(init_type, trans_type, logdensity_type)
end

"""
    initialization(hmm)

Return the vector of initial state probabilities for `hmm`.
"""
function initialization end

"""
    transition_matrix(hmm, control)

Return the matrix of state transition probabilities for `hmm` when `control` is applied.
"""
transition_matrix(hmm::AbstractHMM, control::Nothing) = transition_matrix(hmm)

"""
    obs_distributions(hmm, control)

Return a vector of observation distributions, one for each state of `hmm`  when `control` is applied.

These objects should support

- `rand(rng, dist)`
- `DensityInterface.logdensityof(dist, obs)`
- `StatsAPI.fit!(dist, obs_seq, weight_seq)`
"""
obs_distributions(hmm::AbstractHMM, control::Nothing) = obs_distributions(hmm)

function obs_logdensities!(logb::AbstractVector, hmm::AbstractHMM, obs, control)
    dists = obs_distributions(hmm, control)
    @inbounds for i in eachindex(logb, dists)
        logb[i] = logdensityof(dists[i], obs)
    end
    check_right_finite(logb)
    return nothing
end

"""
    fit!(
        hmm::AbstractHMM,
        obs_seq::AbstractVector;
        control_seq::AbstractVector,
        seq_ends::AbstractVector{Int},
        fb_storage::ForwardBackwardStorage,
    )

Update `hmm` in-place based on information generated during forward-backward.
"""
StatsAPI.fit!  # TODO: complete

## Sampling

"""
    rand([rng,] hmm, T)
    rand([rng,] hmm, control_seq)

Simulate `hmm` for `T` time steps / when the sequence `control_seq` is applied.
    
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

    dists1 = obs_distributions(hmm, control_seq[1])
    obs1 = rand(rng, dists1[state1])
    obs_seq = Vector{typeof(obs1)}(undef, T)
    obs_seq[1] = obs1

    for t in 2:T
        dists = obs_distributions(hmm, control_seq[t])
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
