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
    dists::AbstractVector{<:UnivariateDistribution},
    i::Integer,
    x_nums::AbstractVector,
    w::AbstractVector,
)
    return dists[i] = fit(typeof(dists[i]), x_nums, w)
end

function HiddenMarkovModels.fit_in_sequence!(
    dists::AbstractVector{<:MultivariateDistribution},
    i::Integer,
    x_vecs::AbstractVector{<:AbstractVector},
    w::AbstractVector,
)
    return dists[i] = fit(typeof(dists[i]), reduce(hcat, x_vecs), w)
end

#=

# Matrix distribution fitting not supported by Distributions.jl at the moment

function HiddenMarkovModels.fit_in_sequence!(
    dists::AbstractVector{<:MatrixDistribution},
    i::Integer,
    x_mats::AbstractVector{<:AbstractMatrix},
    w::AbstractVector,
)
    return dists[i] = fit(typeof(dists[i]), reduce(dcat, x_mats), w)
end

dcat(M1, M2) = cat(M1, M2; dims=3)
=#

end
