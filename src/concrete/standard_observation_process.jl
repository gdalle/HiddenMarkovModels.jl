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

distribution(op::StandardObservationProcess, i::Integer) = op.distributions[i]
distributions(op::StandardObservationProcess) = op.distributions

function reestimate!(op::StandardObservationProcess{D}, obs_seq, γ) where {D}
    for i in 1:length(op)
        fit_element_from_sequence!(op.distributions, i, obs_seq, γ[i, :])
    end
end

"""
    fit_element_from_sequence!(dists, i, x, w)

Modify the `i`-th element of `dists` by fitting it to an observation sequence `x` with associated weight sequence `w`.

The default behavior is a fallback on `StatsAPI.fit!(dists[i], x, w)`, which users are encouraged to implement if their observation distributions are mutable.
If not, they should implement `HMMs.fit_element!` instead, as is already done for Distributions.jl in the source code.
"""
function fit_element_from_sequence!(dists, i::Integer, x, w)
    fit!(dists[i], x, w)
    return nothing
end

function fit_element_from_sequence!(
    dists::AbstractVector{D}, i::Integer, x::AbstractVector, w::AbstractVector
) where {D<:UnivariateDistribution}
    dists[i] = fit(D, x, w)
    return nothing
end

function fit_element_from_sequence!(
    dists::AbstractVector{D},
    i::Integer,
    x::AbstractVector{<:AbstractVector},
    w::AbstractVector,
) where {D<:MultivariateDistribution}
    dists[i] = fit(D, reduce(hcat, x), w)
    return nothing
end

function fit_element_from_sequence!(
    dists::AbstractVector{D},
    i::Integer,
    x::AbstractVector{<:AbstractMatrix},
    w::AbstractVector,
) where {D<:MatrixDistribution}
    dists[i] = fit(D, reduce(dcat, x), w)
    return nothing
end

dcat(M1, M2) = cat(M1, M2; dims=3)
