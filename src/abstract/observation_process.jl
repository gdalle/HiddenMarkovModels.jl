"""
    ObservationProcess

Abstract type for the observation part of an HMM.

# Required methods

- `Base.length(op)`
- `distribution(op, i)`

# Optional methods

- `reestimate!(op, obs_seq, γ)`
"""
abstract type ObservationProcess end

## Interface

function Base.length(::OP) where {OP<:ObservationProcess}
    return error("length needs to be implemented for $OP")
end

function distribution(::OP, ::Integer) where {OP<:ObservationProcess}
    return error("distribution(op, i) needs to be implemented for $OP")
end

## Fallbacks

function distributions(op::ObservationProcess)
    return [distribution(op, i) for i in 1:length(op)]
end

## Checks

function check(op::ObservationProcess)
    N = length(op)
    if !(N > 0)
        throw(ArgumentError("No states in observation process"))
    else
        for i in 1:N
            if DensityKind(distribution(op, i)) == NoDensity()
                err_msg = "Observation process does not satisfy DensityInterface.jl"
                throw(ArgumentError(err_msg))
            end
        end
    end
end

## Simulation

function Base.rand(rng::AbstractRNG, op::ObservationProcess, state_seq)
    obs_seq = [rand(rng, distribution(op, i)) for i in state_seq]
    return obs_seq
end

function Base.rand(op::ObservationProcess, state_seq)
    return rand(GLOBAL_RNG, op, state_seq)
end
