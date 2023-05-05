"""
    AbstractHiddenMarkovModel

Interface for Hidden Markov Models with arbitrary emissions.

# Required methods

- [`nb_states(hmm, θ)`](@ref)
- [`initial_distribution(hmm, θ)`](@ref)
- [`transition_matrix(hmm, θ)`](@ref)
- [`emission_distribution(hmm, i, θ)`](@ref)
"""
abstract type AbstractHiddenMarkovModel end

"""
    AbstractHMM

Shortcut for [`AbstractHiddenMarkovModel`](@ref).
"""
const AbstractHMM = AbstractHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractHMM) = HasDensity()

"""
    nb_states(hmm, θ)

Return the number of states of `hmm` with parameters `θ`.
"""
function nb_states end

"""
    initial_distribution(hmm, θ)

Return the vector of initial state probabilities for `hmm` with parameters `θ`.
"""
function initial_distribution end

"""
    transition_matrix(hmm, θ)

Return the matrix of state transition probabilities for `hmm` with parameters `θ`.
"""
function transition_matrix end

"""
    emission_distribution(hmm, θ, i)

Return the emission distribution in state `i` for `hmm` with parameters `θ`.

The result must satisfy the DensityInterface.jl specification.
"""
function emission_distribution end

"""
    emission_distributions(hmm, θ)

Return the vector of emission distributions for `hmm` with parameters `θ`.

Each element of the result result must satisfy the DensityInterface.jl specification.
"""
function emission_distributions(hmm::AbstractHMM, θ)
    return [emission_distribution(hmm, θ, i) for i in 1:nb_states(hmm)]
end
