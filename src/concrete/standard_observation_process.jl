"""
    StandardObservationProcess{D} <: ObservationProcess

# Fields

- `distributions::AbstractVector{D}`: one distribution per state
"""
struct StandardObservationProcess{D,V<:AbstractVector{D}} <: ObservationProcess
    distributions::V
end

function Base.copy(op::StandardObservationProcess)
    return StandardObservationProcess(copy(op.distributions))
end

function Base.show(io::IO, op::StandardObservationProcess{D,V}) where {D,V}
    return print(io, "StandardObservationProcess{$D,$V} with $(length(op)) states")
end

Base.length(op::StandardObservationProcess) = length(op.distributions)

distribution(op::StandardObservationProcess, i::Integer) = op.distributions[i]
distributions(op::StandardObservationProcess) = op.distributions

function reestimate!(op::StandardObservationProcess{D}, obs_seq, γ) where {D}
    for i in 1:length(op)
        @views op.distributions[i] = fit(D, obs_seq, γ[i, :])
    end
end
