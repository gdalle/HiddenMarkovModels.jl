"""
    AbstractHiddenMarkovModel 

Abstract supertype for an HMM amenable to simulation, inference and learning.

# Interface

To create your own subtype of `AbstractHiddenMarkovModel`, you need to implement the following methods:

- [`length(hmm)`](@ref)
- [`eltype(hmm, obs)`](@ref)
- [`initialization(hmm)`](@ref)
- [`transition_matrix(hmm)`](@ref)
- [`obs_logdensities!(logb, hmm, obs)`](@ref)
- [`obs_sample(rng, hmm, i)`](@ref) (optional)
- [`fit!(hmm, init_count, trans_count, obs_seq, state_marginals)`](@ref) (optional)

# Applicable functions

Any HMM object which satisfies the interface can be given as input to the following functions:

- [`logdensityof(hmm, obs_seq)`](@ref)
- [`forward(hmm, obs_seq)`](@ref)
- [`viterbi(hmm, obs_seq)`](@ref)
- [`forward_backward(hmm, obs_seq)`](@ref)
- [`rand(rng, hmm, T)`](@ref) (if `obs_sample(rng, hmm, i)` is implemented)
- [`baum_welch(hmm, obs_seq)`](@ref) (if `fit!` is implemented)
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
Base.eltype

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
    obs_logdensities!(logb, hmm, obs)

Fill the vector `logb` with the logdensities of observation `obs` for each state of `hmm`.
"""
function obs_logdensities! end

"""
    obs_sample(rng, hmm, i)

Sample from the observation distribution of `hmm` corresponding to state `i`.
"""
function obs_sample end

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

## Fallbacks

function obs_logdensities!(logB::AbstractMatrix, hmm::AbstractHMM, obs_seq::Vector)
    for (logb, obs) in zip(eachcol(logB), obs_seq)
        obs_logdensities!(logb, hmm, obs)
    end
end

function obs_logdensities_matrix(hmm::AbstractHMM, obs_seq::Vector)
    R = eltype(hmm, obs_seq[1])
    logB = Matrix{R}(length(hmm), length(obs_seq))
    obs_logdensities!(logB, hmm, obs_seq)
    return logB
end

## Sampling

"""
    rand(hmm, T)
    rand(rng, hmm, T)

Simulate `hmm` for `T` time steps. 
"""
function Base.rand(rng::AbstractRNG, hmm::AbstractHMM, T::Integer)
    p = initialization(hmm)
    A = transition_matrix(hmm)
    state_seq = Vector{Int}(undef, T)
    state_seq[1] = rand(rng, Categorical(p; check_args=false))
    @views for t in 2:T
        state_seq[t] = rand(rng, Categorical(A[state_seq[t - 1], :]; check_args=false))
    end
    obs_seq = [obs_sample(rng, hmm, state_seq[t]) for t in 1:T]
    return (; state_seq=state_seq, obs_seq=obs_seq)
end

Base.rand(hmm::AbstractHMM, T::Integer) = rand(default_rng(), hmm, T)
