"""
    AbstractHiddenMarkovModel <: AbstractMarkovChain 

Abstract supertype for an HMM amenable to simulation, inference and learning.

# Required interface

- `initial_distribution(hmm)`
- `transition_matrix(hmm)`
- `obs_distribution(hmm, i)`
- `fit!(hmm, obs_seqs, fbs)` (optional)

# Applicable methods

- `rand([rng,] hmm, T)`
- `logdensityof(hmm, obs_seq)` / `logdensityof(hmm, obs_seqs, nb_seqs)`
- `forward(hmm, obs_seq)` / `forward(hmm, obs_seqs, nb_seqs)`
- `viterbi(hmm, obs_seq)` / `viterbi(hmm, obs_seqs, nb_seqs)`
- `forward_backward(hmm, obs_seq)` / `forward_backward(hmm, obs_seqs, nb_seqs)`
- `baum_welch(hmm, obs_seq)` / `baum_welch(hmm, obs_seqs, nb_seqs)` if `fit!` is implemented
"""
abstract type AbstractHiddenMarkovModel <: AbstractMarkovChain end

"""
    AbstractHMM

Alias for the type `AbstractHiddenMarkovModel`.
"""
const AbstractHMM = AbstractHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractHMM) = HasDensity()

@required AbstractHMM begin
    Base.length(::AbstractHMM)
    initial_distribution(::AbstractHMM)
    transition_matrix(::AbstractHMM)
    obs_distribution(::AbstractHMM, ::Integer)
end

"""
    obs_distribution(hmm::AbstractHMM, i)

Return the observation distribution of `hmm` associated with state `i`.

The returned object `dist` must implement
- `rand(rng, dist)`
- `DensityInterface.logdensityof(dist, x)`
"""
function obs_distribution end

"""
    StatsAPI.fit!(hmm::AbstractHMM, obs_seqs, fbs)
"""
StatsAPI.fit!  # TODO: docstring

function Base.rand(rng::AbstractRNG, hmm::AbstractHMM, T::Integer)
    mc = MarkovChain(hmm)
    state_seq = rand(rng, mc, T)
    first_obs = rand(rng, obs_distribution(hmm, first(state_seq)))
    obs_seq = Vector{typeof(first_obs)}(undef, T)
    obs_seq[1] = first_obs
    for t in 2:T
        obs_seq[t] = rand(rng, obs_distribution(hmm, state_seq[t]))
    end
    return (; state_seq=state_seq, obs_seq=obs_seq)
end

function MarkovChain(hmm::AbstractHMM)
    return MarkovChain(initial_distribution(hmm), transition_matrix(hmm))
end
