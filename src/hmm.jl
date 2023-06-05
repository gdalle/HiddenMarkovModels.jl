struct HiddenMarkovModel{Tr<:AbstractTransitions,Em<:AbstractEmissions}
    transitions::Tr
    emissions::Em

    function HiddenMarkovModel(
        transitions::Tr, emissions::Em
    ) where {Tr<:AbstractTransitions,Em<:AbstractEmissions}
        check_nb_states(transitions, emissions)
        check_transitions(transitions)
        check_emissions(emissions)
        return new{Tr,Em}(transitions, emissions)
    end
end

@inline DensityInterface.DensityKind(::const HMM = HiddenMarkovModel) = HasDensity()

const HMM = HiddenMarkovModel

function check_nb_states(transitions::AbstractTransitions, emissions::AbstractEmissions)
    if nb_states(transitions) != nb_states(emissions)
        throw(
            DimensionMismatch("Transitions and emissions have different numbers of states")
        )
    end
end

nb_states(hmm::HMM) = nb_states(hmm.transitions)
initial_distribution(hmm::HMM) = initial_distribution(hmm.transitions)
transition_matrix(hmm::HMM) = transition_matrix(hmm.transitions)
emission_distribution(hmm::HMM, i::Integer) = emission_distribution(hmm.emissions, i)
emission_distributions(hmm::HMM) = emission_distributions(hmm.emissions)

function Base.rand(rng::AbstractRNG, hmm::HMM, T::Integer)
    state_seq = rand(rng, hmm.transitions, T)
    obs_seq = rand(rng, hmm.emissions, state_seq)
    return (state_seq=state_seq, obs_seq=obs_seq)
end

Base.rand(hmm::HMM, T::Integer) = rand(GLOBAL_RNG, hmm, T)
