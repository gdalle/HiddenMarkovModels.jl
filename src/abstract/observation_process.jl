"""
    ObservationProcess

Abstract type for the observation part of an HMM.

# Required methods

- `length(op)`
- `obs_distribution(op, i)`

# Optional methods

- `fit!(op, obs_seq, γ)`
"""
abstract type ObservationProcess end

## Interface

"""
    length(op::ObservationProcess)

Return the number of states of `op`.
"""
function Base.length(::OP) where {OP<:ObservationProcess}
    return error("$OP needs to implement Base.length")
end

"""
    obs_distribution(op::ObservationProcess, i)

Return the observation distribution of `op` associated with state `i`.
"""
function obs_distribution(::OP, ::Integer) where {OP<:ObservationProcess}
    return error("$OP needs to implement HMMs.obs_distribution(op, i)")
end

"""
    StatsAPI.fit!(op::ObservationProcess, obs_seq, γ)

Update all observation distributions of `op` based on an observation sequence `obs_seq`, weighted by `γ[i, :]` for each state `i`.
"""
function StatsAPI.fit!(::OP, obs_seq, γ) where {OP<:ObservationProcess}
    return error("$OP needs to implement StatsAPI.fit!(op, obs_seq, γ) for Baum-Welch.")
end

## Checks

function check(op::ObservationProcess)
    N = length(op)
    if !(N > 0)
        throw(ArgumentError("No states in observation process"))
    else
        for i in 1:N
            if DensityKind(obs_distribution(op, i)) == NoDensity()
                err_msg = "Observation process does not satisfy DensityInterface.jl"
                throw(ArgumentError(err_msg))
            end
        end
    end
end

## Simulation

function Base.rand(rng::AbstractRNG, op::ObservationProcess, state_seq)
    obs_seq = [rand(rng, obs_distribution(op, i)) for i in state_seq]
    return obs_seq
end

function Base.rand(op::ObservationProcess, state_seq)
    return rand(GLOBAL_RNG, op, state_seq)
end
