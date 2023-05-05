struct HiddenMarkovModel{T1,T2,D} <: AbstractHMM
    initial_distribution::Vector{T1}
    transition_matrix::Matrix{T2}
    emission_distributions::Vector{D}

    function HiddenMarkovModel{T1,T2,D}(
        initial_distribution::Vector{T1},
        transition_matrix::Matrix{T2},
        emission_distributions::Vector{D},
    ) where {T1,T2,D}
        check_hmm_args(initial_distribution, transition_matrix, emission_distributions)
        return new{T1,T2,D}(initial_distribution, transition_matrix, emission_distributions)
    end
end

const HMM = HiddenMarkovModel

function HiddenMarkovModel(
    initial_distribution::Vector{T1},
    transition_matrix::Matrix{T2},
    emission_distributions::Vector{D},
) where {T1,T2,D}
    return HiddenMarkovModel{T1,T2,D}(
        initial_distribution, transition_matrix, emission_distributions
    )
end

function check_hmm_args(initial_distribution, transition_matrix, emission_distributions)
    N = length(initial_distribution)
    if N <= 0
        throw(ArgumentError("Initial distribution has no states"))
    elseif size(transition_matrix) != (N, N)
        throw(
            DimensionMismatch(
                "Initial distribution and transition matrix don't have the same number of states",
            ),
        )
    elseif length(emission_distributions) != N
        throw(
            DimensionMismatch(
                "Initial distribution and emissions don't have the same number of states"
            ),
        )
    elseif !is_prob_vec(initial_distribution)
        throw(ArgumentError("Initial distribution is not valid"))
    elseif !is_trans_mat(transition_matrix)
        throw(ArgumentError("Transition matrix is not valid"))
    elseif DensityKind(first(emission_distributions)) == NoDensity()
        throw(
            ArgumentError("Emissions do not satisfy the DensityInterface.jl specification")
        )
    end
end

function nb_states(hmm::HMM, θ)
    return length(hmm.initial_distribution)
end

function initial_distribution(hmm::HMM, θ)
    return hmm.initial_distribution
end

function transition_matrix(hmm::HMM, θ)
    return hmm.transition_matrix
end

function emission_distribution(hmm::HMM, θ, i)
    return hmm.emission_distributions[i]
end

function emission_distributions(hmm::HMM, θ)
    return hmm.emission_distributions
end

function emission_type(hmm::HMM{T1,T2,D}) where {T1,T2,D}
    return D
end
