"""
    AbstractHiddenMarkovModel 

Abstract supertype for an HMM amenable to simulation, inference and learning.

# Required interface

- `initial_distribution(hmm)`
- `transition_matrix(hmm)`
- `obs_distribution(hmm, i)`
- `fit!(hmm, init_count, trans_count, obs_seq, state_marginals)` (optional)

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

"""
    length(hmm::AbstractHMM) 

Return the number of states of `hmm`.
"""
Base.length

"""
    initial_distribution(hmm::AbstractHMM)

Return the initial state probabilities of `hmm`.
"""
function initial_distribution end

"""
    transition_matrix(hmm::AbstractHMM) 

Return the state transition probabilities of `hmm`.
"""
function transition_matrix end

"""
    obs_distribution(hmm::AbstractHMM, i)

Return the observation distribution of `hmm` associated with state `i`.

The returned object `dist` must implement
- `rand(rng, dist)`
- `DensityInterface.logdensityof(dist, x)`
"""
function obs_distribution end

## Sampling

function Base.rand(rng::AbstractRNG, hmm::AbstractHMM, T::Integer)
    init = initial_distribution(hmm)
    trans = transition_matrix(hmm)
    first_state = rand(rng, Categorical(init; check_args=false))
    state_seq = Vector{Int}(undef, T)
    state_seq[1] = first_state
    @views for t in 2:T
        state_seq[t] = rand(rng, Categorical(trans[state_seq[t - 1], :]; check_args=false))
    end
    first_obs = rand(rng, obs_distribution(hmm, first(state_seq)))
    obs_seq = Vector{typeof(first_obs)}(undef, T)
    obs_seq[1] = first_obs
    for t in 2:T
        obs_seq[t] = rand(rng, obs_distribution(hmm, state_seq[t]))
    end
    return (; state_seq=state_seq, obs_seq=obs_seq)
end

Base.rand(hmm::AbstractHMM, T::Integer) = rand(default_rng(), hmm, T)
