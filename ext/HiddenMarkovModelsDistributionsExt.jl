module HiddenMarkovModelsDistributionsExt

using HiddenMarkovModels: HiddenMarkovModels
using Distributions:
    Distributions,
    Distribution,
    UnivariateDistribution,
    MultivariateDistribution,
    MatrixDistribution,
    fit

function HiddenMarkovModels.fit_in_sequence!(
    dists::AbstractVector{D}, i::Integer, x_nums::AbstractVector, w::AbstractVector
) where {D<:UnivariateDistribution}
    return dists[i] = fit(D, x_nums, w)
end

function HiddenMarkovModels.fit_in_sequence!(
    dists::AbstractVector{D},
    i::Integer,
    x_vecs::AbstractVector{<:AbstractVector},
    w::AbstractVector,
) where {D<:MultivariateDistribution}
    return dists[i] = fit(D, reduce(hcat, x_vecs), w)
end

function HiddenMarkovModels.fit_in_sequence!(
    dists::AbstractVector{D},
    i::Integer,
    x_mats::AbstractVector{<:AbstractMatrix},
    w::AbstractVector,
) where {D<:MatrixDistribution}
    return dists[i] = fit(D, reduce(dcat, x_mats), w)
end

dcat(M1, M2) = cat(M1, M2; dims=3)

end
