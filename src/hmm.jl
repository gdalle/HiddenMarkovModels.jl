"""
    HiddenMarkovModel{SP<:StateProcess,OP<:ObservationProcess}

Combination of a state and an observation process, amenable to simulation, inference and learning.

# Fields

- `state_process::SP`
- `obs_process::OP`

Both fields are considered part of the API, which is why there are no accessors.

# Applicable methods

- `initial_distribution(hmm)`
- `transition_matrix(hmm)`
- `obs_distribution(hmm, i)`
- `rand([rng,] hmm, T)`
- `logdensityof(hmm, obs_seq)`
- `viterbi(hmm, obs_seq)`
- `forward_backward(hmm, obs_seq)`
- `baum_welch(hmm, obs_seq)`
"""
struct HiddenMarkovModel{SP<:StateProcess,OP<:ObservationProcess}
    state_process::SP
    obs_process::OP

    function HiddenMarkovModel(
        state_process::SP, obs_process::OP
    ) where {SP<:StateProcess,OP<:ObservationProcess}
        if length(state_process) != length(obs_process)
            msg = "State process and observation process have different numbers of states"
            throw(DimensionMismatch(msg))
        end
        check(state_process)
        check(obs_process)
        return new{SP,OP}(state_process, obs_process)
    end
end

"""
    HMM

Alias for the struct `HiddenMarkovModel`.
"""
const HMM = HiddenMarkovModel

@inline DensityInterface.DensityKind(::HMM) = HasDensity()

## Short constructor

"""
    HMM(p, A, dists)

Construct an HMM from a vector of initial probabilities, a matrix of transition probabilities and a vector of observation distributions.

Same constructor as in HMMBase.jl.
"""
function HiddenMarkovModel(p::AbstractVector, A::AbstractMatrix, dists::AbstractVector)
    sp = StandardStateProcess(p, A)
    op = StandardObservationProcess(dists)
    return HiddenMarkovModel(sp, op)
end

## Base utils

function Base.copy(hmm::HMM)
    return HiddenMarkovModel(copy(hmm.state_process), copy(hmm.obs_process))
end

function Base.show(io::IO, hmm::HMM{SP,OP}) where {SP,OP}
    print(io, "HiddenMarkovModel{$SP,$OP}")
    return nothing
end

"""
    length(hmm::HMM)

Return the number of states of `hmm`.
"""
Base.length(hmm::HMM) = length(hmm.state_process)

## Inherited interface (not used in the package code)

"""
    initial_distribution(hmm::HMM)

Return the initial state probabilities of `hmm`.
"""
initial_distribution(hmm::HMM) = initial_distribution(hmm.state_process)

"""
    transition_matrix(hmm::HMM)

Return the state transition probabilities of `hmm`.
"""
transition_matrix(hmm::HMM) = transition_matrix(hmm.state_process)

"""
    obs_distribution(hmm::HMM, i)

Return the observation distribution of `hmm` associated with state `i`.
"""
obs_distribution(hmm::HMM, i::Integer) = obs_distribution(hmm.obs_process, i)

## Simulation

"""
    rand(rng, hmm, T)

Simulate an HMM for `T` time steps with a specified `rng`.
"""
function Base.rand(rng::AbstractRNG, hmm::HMM, T::Integer)
    state_seq = rand(rng, hmm.state_process, T)
    obs_seq = rand(rng, hmm.obs_process, state_seq)
    return (state_seq=state_seq, obs_seq=obs_seq)
end

"""
    rand(hmm, T)

Simulate an HMM for `T` time steps.
"""
Base.rand(hmm::HMM, T::Integer) = rand(GLOBAL_RNG, hmm, T)
