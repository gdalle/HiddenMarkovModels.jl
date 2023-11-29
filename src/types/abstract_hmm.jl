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
- [`fit!(hmm, init_count, trans_count, obs_seq, state_marginals)`](@ref) (optional)

# Applicable functions

Any HMM object which satisfies the interface can be given as input to the following functions:

- [`rand(rng, hmm, T)`](@ref)
- [`logdensityof(hmm, obs_seq)`](@ref)
- [`forward(hmm, obs_seq)`](@ref)
- [`viterbi(hmm, obs_seq)`](@ref)
- [`forward_backward(hmm, obs_seq)`](@ref)
- [`baum_welch(hmm, obs_seq)`](@ref) (if `fit!` is implemented)
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
    d = obs_distributions(hmm, t)
    @inbounds for i in eachindex(logb, d)
        logb[i] = logdensityof(d[i], obs)
    end
end

"""
    fit!(hmm, init_count, trans_count, obs_seq, state_marginals)

Update `hmm` in-place based on information generated during forward-backward.

This method is only necessary for the Baum-Welch algorithm.

# Arguments

- `init_count::Vector`: posterior initialization counts for each state (size `N`)
- `trans_count::AbstractMatrix`: posterior transition counts for each state (size `(N, N)`)
- `obs_seq::Vector`: sequence of observation, possibly concatenated (size `T`)
- `state_marginals::Matrix`: posterior probabilities of being in each state at each time, to be used as weights during maximum likelihood fitting of the observation distributions (size `(N, T)`).

# See also

- [`BaumWelchStorage`](@ref)
- [`ForwardBackwardStorage`](@ref)
"""
StatsAPI.fit!  # TODO: complete

## Sampling

"""
    rand(hmm, T)
    rand(rng, hmm, T)

Simulate `hmm` for `T` time steps. 
"""
function Base.rand(rng::AbstractRNG, hmm::AbstractHMM, T::Integer)
    dummy_log_probas = fill(-Inf, length(hmm))

    p = initialization(hmm)
    state1 = rand(rng, LightCategorical(p, dummy_log_probas))
    state_seq = Vector{typeof(state1)}(undef, T)
    state_seq[1] = state1

    d = obs_distributions(hmm, 1)
    obs1 = rand(rng, d[state1])
    obs_seq = Vector{typeof(obs1)}(undef, T)
    obs_seq[1] = obs1

    @views for t in 2:T
        A = transition_matrix(hmm, t)
        d = obs_distributions(hmm, t)
        state_seq[t] = rand(rng, LightCategorical(A[state_seq[t - 1], :], dummy_log_probas))
        obs_seq[t] = rand(rng, d[state_seq[t]])
    end
    return (; state_seq=state_seq, obs_seq=obs_seq)
end

Base.rand(hmm::AbstractHMM, T::Integer) = rand(default_rng(), hmm, T)
