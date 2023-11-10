"""
    AbstractHiddenMarkovModel 

Abstract supertype for an HMM amenable to simulation, inference and learning.

# Interface

To create your own subtype of `AbstractHiddenMarkovModel`, you need to implement the following methods:

- [`length(hmm)`](@ref)
- [`eltype(hmm, obs)`](@ref)
- [`initialization(hmm)`](@ref)
- [`transition_matrix(hmm)`](@ref)
- [`obs_distributions(hmm)`](@ref)
- [`fit!(hmm, init_count, trans_count, obs_seq, state_marginals)`](@ref) (optional)

# Applicable functions

Any HMM object which satisfies the interface can be given as input to the following functions:

- [`rand(rng, hmm, T)`](@ref)
- [`logdensityof(hmm, obs_seq)`](@ref)
- [`forward(hmm, obs_seq)`](@ref)
- [`viterbi(hmm, obs_seq)`](@ref)
- [`forward_backward(hmm, obs_seq)`](@ref)
- [`baum_welch(hmm, obs_seq)`](@ref) (if the optional `fit!` is implemented)
"""
abstract type AbstractHiddenMarkovModel end

"""
    AbstractHMM

Alias for the type `AbstractHiddenMarkovModel`.
"""
const AbstractHMM = AbstractHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractHMM) = HasDensity()

## Interface

"""
    length(hmm)

Return the number of states of `hmm`.
"""
Base.length

"""
    eltype(hmm, obs)

Return a type that can accommodate forward-backward computations on observations similar to `obs`.
It is typicall a promotion between the element type of the initialization, the element type of the transition matrix, and the type of an observation logdensity evaluated at `obs`.
"""
function Base.eltype(hmm::AbstractHMM, obs)
    init_type = eltype(initialization(hmm))
    trans_type = eltype(transition_matrix(hmm))
    logdensity_type = typeof(logdensityof(obs_distributions(hmm)[1], obs))
    return promote_type(init_type, trans_type, logdensity_type)
end

"""
    initialization(hmm)

Return the vector of initial state probabilities for `hmm`.
"""
function initialization end

"""
    transition_matrix(hmm) 

Return the matrix of state transition probabilities for `hmm`.
"""
function transition_matrix end

"""
    obs_distributions(hmm)

Return a vector of observation distributions for `hmm`.

Each element `dist` of this vector must implement
- `rand(rng, dist)`
- `DensityInterface.logdensityof(dist, obs)`
"""
function obs_distributions end

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
    p = initialization(hmm)
    A = transition_matrix(hmm)
    d = obs_distributions(hmm)

    first_state = rand(rng, Categorical(p; check_args=false))
    state_seq = Vector{Int}(undef, T)
    state_seq[1] = first_state
    @views for t in 2:T
        state_seq[t] = rand(rng, Categorical(A[state_seq[t - 1], :]; check_args=false))
    end
    first_obs = rand(rng, d[state_seq[1]])
    obs_seq = Vector{typeof(first_obs)}(undef, T)
    obs_seq[1] = first_obs
    for t in 2:T
        obs_seq[t] = rand(rng, d[state_seq[t]])
    end
    return (; state_seq=state_seq, obs_seq=obs_seq)
end

Base.rand(hmm::AbstractHMM, T::Integer) = rand(default_rng(), hmm, T)
