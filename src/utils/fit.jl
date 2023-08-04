"""
    fit_element_from_sequence!(dists, i, x, w)

Modify the `i`-th element of `dists` by fitting it to an observation sequence `x` with associated weight sequence `w`.

The default behavior is a fallback on `StatsAPI.fit!`, which users are encouraged to implement if their observation distributions are mutable.
If this is not possible, please override `fit_element_from_sequence!` directly.
"""
function fit_element_from_sequence!(dists, i, x, w)
    fit!(dists[i], x, w)
    return nothing
end

function fit_element_from_sequence!(
    dists::AbstractVector{D}, i::Integer, x::AbstractVector, w::AbstractVector
) where {D<:Distribution}
    dists[i] = _fit_from_sequence(D, x, w)
    return nothing
end

function _fit_from_sequence(
    ::Type{D}, x_nums::AbstractVector, w::AbstractVector
) where {D<:UnivariateDistribution}
    return fit(D, x_nums, w)
end

function _fit_from_sequence(
    ::Type{D}, x_vecs::AbstractVector{<:AbstractVector}, w::AbstractVector
) where {D<:MultivariateDistribution}
    return fit(D, reduce(hcat, x_vecs), w)
end

function _fit_from_sequence(
    ::Type{D}, x_mats::AbstractVector{<:AbstractMatrix}, w::AbstractVector
) where {D<:MatrixDistribution}
    return fit(D, reduce(dcat, x_mats), w)
end

dcat(M1, M2) = cat(M1, M2; dims=3)
