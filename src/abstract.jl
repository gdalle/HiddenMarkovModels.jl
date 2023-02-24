"""
    AbstractHiddenMarkovModel

Interface for Hidden Markov Models with arbitrary emissions.

# Required methods

- [`nb_states(hmm)`](@ref)
- [`initial_distribution(hmm)`](@ref)
- [`transition_matrix(hmm)`](@ref)
- [`emission_distribution(hmm, i)`](@ref)
"""
abstract type AbstractHiddenMarkovModel end

"""
    AbstractHMM

Shortcut for [`AbstractHiddenMarkovModel`](@ref).
"""
const AbstractHMM = AbstractHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractHMM) = HasDensity()

"""
    nb_states(hmm)

Return the number of states of `hmm`.
"""
function nb_states end

"""
    initial_distribution(hmm)

Return the vector of initial state probabilities for `hmm`.
"""
function initial_distribution end

"""
    transition_matrix(hmm)

Return the matrix of state transition probabilities for `hmm`.
"""
function transition_matrix end

"""
    emission_distribution(hmm, i)

Return the distribution of emission in state `i` for `hmm` as an object satisfying the DensityInterface.jl specification.
"""
function emission_distribution end
function emission_distributions end
