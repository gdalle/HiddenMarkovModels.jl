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
    print(io, "StandardObservationProcess{$D,$V}")
    return nothing
end

Base.length(op::StandardObservationProcess) = length(op.distributions)

obs_distribution(op::StandardObservationProcess, i::Integer) = op.distributions[i]

function StatsAPI.fit!(op::StandardObservationProcess{D}, obs_seq, γ) where {D}
    for i in 1:length(op)
        fit_element_from_sequence!(op.distributions, i, obs_seq, γ[i, :])
    end
end
