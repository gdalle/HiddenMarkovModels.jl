module HiddenMarkovModelsDistributionsExt

using HiddenMarkovModels: HiddenMarkovModels
using Distributions:
    Distributions,
    Distribution,
    UnivariateDistribution,
    MultivariateDistribution,
    MatrixDistribution

function HiddenMarkovModels.fit_element_from_sequence!(
    dists::AbstractVector{D}, i::Integer, x, w
) where {D<:Distribution}
    dists[i] = fit_from_sequence(D, x, w)
    return nothing
end

function fit_from_sequence(
    ::Type{D}, x_nums::AbstractVector, w::AbstractVector
) where {D<:UnivariateDistribution}
    return fit(D, x_nums, w)
end

function fit_from_sequence(
    ::Type{D}, x_vecs::AbstractVector{<:AbstractVector}, w::AbstractVector
) where {D<:MultivariateDistribution}
    return fit(D, reduce(hcat, x_vecs), w)
end

function fit_from_sequence(
    ::Type{D}, x_mats::AbstractVector{<:AbstractMatrix}, w::AbstractVector
) where {D<:MatrixDistribution}
    return fit(D, reduce(dcat, x_mats), w)
end

dcat(M1, M2) = cat(M1, M2; dims=3)

end
