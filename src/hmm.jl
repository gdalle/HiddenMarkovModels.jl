struct HiddenMarkovModel{T1,T2,D} <: AbstractHMM
    initial_distribution::Vector{T1}
    transition_matrix::Matrix{T2}
    emission_distributions::Vector{D}

    function HiddenMarkovModel(
        initial_distribution::Vector{T1},
        transition_matrix::Matrix{T2},
        emission_distributions::Vector{D},
    ) where {T1,T2,D}
        N = length(initial_distribution)
        @assert size(transition_matrix) == (N, N)
        @assert length(emission_distributions) == N
        return new{T1,T2,D}(initial_distribution, transition_matrix, emission_distributions)
    end
end

const HMM = HiddenMarkovModel

function nb_states(hmm::HMM)
    return length(hmm.initial_distribution)
end

function initial_distribution(hmm::HMM)
    return hmm.initial_distribution
end

function transition_matrix(hmm::HMM)
    return hmm.transition_matrix
end

function emission_distribution(hmm::HMM, i)
    return hmm.emission_distributions[i]
end

function emission_distributions(hmm::HMM)
    return hmm.emission_distributions
end
