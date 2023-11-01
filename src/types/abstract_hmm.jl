"""
    AbstractHiddenMarkovModel 

Abstract supertype for an HMM amenable to simulation, inference and learning.

# Required interface

- `initialization(hmm)`
- `transition_matrix(hmm)`
- `obs_distributions(hmm)`

# Optional interface

- `transition_matrix!(trans, hmm, t)`
- `obs_distributions!(dists, hmm, t)`
- `fit!(hmm, obs_seqs, fbs)`

# Applicable methods

- `rand([rng,] hmm, T)`
- `logdensityof(hmm, obs_seq)` / `logdensityof(hmm, obs_seqs, nb_seqs)`
- `forward(hmm, obs_seq)` / `forward(hmm, obs_seqs, nb_seqs)`
- `viterbi(hmm, obs_seq)` / `viterbi(hmm, obs_seqs, nb_seqs)`
- `forward_backward(hmm, obs_seq)` / `forward_backward(hmm, obs_seqs, nb_seqs)`
- `baum_welch(hmm, obs_seq)` / `baum_welch(hmm, obs_seqs, nb_seqs)` if `fit!` is implemented
"""
abstract type AbstractHiddenMarkovModel end

"""
    AbstractHMM

Alias for the type `AbstractHiddenMarkovModel`.
"""
const AbstractHMM = AbstractHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractHMM) = HasDensity()

## Interface

@required AbstractHMM begin
    Base.length(::AbstractHMM)
    initialization(::AbstractHMM)
    transition_matrix(::AbstractHMM)
    obs_distributions(::AbstractHMM)
end

"""
    length(hmm::AbstractHMM) 

Return the number of states of `hmm`.
"""
Base.length

"""
    initialization(hmm::AbstractHMM)

Return the initial state probabilities of `hmm`.
"""
function initialization end

"""
    transition_matrix(hmm::AbstractHMM) 

Return the state transition probabilities of `hmm`.
"""
function transition_matrix end
transition_matrix!(trans::AbstractMatrix, hmm::AbstractHMM, t::Integer) = nothing

"""
    obs_distribution(hmm::AbstractHMM)

Return a vector of observation distributions of `hmm`, one associated with each state.

These distributions must implement
- `rand(rng, dist)`
- `DensityInterface.logdensityof(dist, x)`
"""
function obs_distributions end
obs_distributions!(dists::AbstractVector, hmm::AbstractHMM, t::Integer) = nothing

"""
    StatsAPI.fit!(hmm::AbstractHMM, obs_seqs, fbs)

Modify `hmm` by estimating new parameters based on observation sequences `obs_seqs` and a vector `fbs` of `ForwardBackwardStorage` objects.
"""
StatsAPI.fit!  # TODO: docstring

## Simulation

function Base.rand(rng::AbstractRNG, hmm::AbstractHMM, T::Integer)
    # Parameters
    init = initialization(hmm)
    trans = transition_matrix(hmm)
    dists = obs_distributions(hmm)
    # Storage
    state_seq = Vector{Int}(undef, T)
    obs_seq = Vector{typeof(rand(rng, dists[1]))}(undef, T)
    # States
    first_state = rand(rng, Categorical(init; check_args=false))
    state_seq[1] = first_state
    for t in 1:T
        transition_matrix!(trans, hmm, t)
        next_state_dist = Categorical(view(trans, state_seq[t], :); check_args=false)
        state_seq[t + 1] = rand(rng, next_state_dist)
    end
    # Observations
    for t in 1:T
        obs_distributions!(dists, hmm, t)
        obs_seq[t] = rand(rng, dists[state_seq[t]])
    end
    return (; state_seq=state_seq, obs_seq=obs_seq)
end
