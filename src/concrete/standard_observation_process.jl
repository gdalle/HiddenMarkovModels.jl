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
    print(io, "StandardObservationProcess:")
    for dist in op.distributions
        print(io, "\n    $dist")
    end
    return nothing
end

Base.length(op::StandardObservationProcess) = length(op.distributions)

distribution(op::StandardObservationProcess, i::Integer) = op.distributions[i]
distributions(op::StandardObservationProcess) = op.distributions

function reestimate!(op::StandardObservationProcess{D}, obs_seq, γ) where {D}
    for i in 1:length(op)
        @views op.distributions[i] = fit_from_sequence(D, obs_seq, γ[i, :])
    end
end

"""
    fit_from_sequence(::Type{D}, x, w)

Fit a distribution of type `D` based on a single sequence of observations `x` associated with a single sequence of weights `w`.

Default to `StatsAPI.fit`, with a special case for Distributions.jl and vectors of vectors (because this implementation of `fit` accepts matrices instead).
Users are free to override this default for concrete distributions.
"""
function fit_from_sequence(::Type{D}, x::AbstractVector, w::AbstractVector) where {D}
    return fit(D, x, w)
end

function fit_from_sequence(
    ::Type{D}, x::AbstractVector{<:AbstractVector}, w::AbstractVector
) where {D<:MultivariateDistribution}
    return fit(D, reduce(hcat, x), w)
end

function reestimate!(::SP, p_count, A_count) where {SP<:StateProcess}
    return error("$SP needs to implement reestimate!(sp, p_count, A_count) for Baum-Welch.")
end

function reestimate!(::OP, obs_seq, γ) where {OP<:ObservationProcess}
    return error("$OP needs to implement reestimate!(op, obs_seq, γ) for Baum-Welch.")
end
