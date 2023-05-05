
struct MyNormal{T1,T2}
    μ::T1
    σ²::T2
end

@inline DensityInterface.DensityKind(::MyNormal) = HasDensity()

function Base.rand(rng::AbstractRNG, dist::MyNormal)
    return sqrt(dist.σ²) * randn(rng) + dist.μ
end

function DensityInterface.densityof(dist::MyNormal, x)
    return inv(sqrt(2π * dist.σ²)) * exp(-(x - dist.μ)^2 * inv(2 * dist.σ²))
end

function fit_mle_from_multiple_sequences(::Type{<:MyNormal}, xs, ws)
    n = sum(length, xs)
    μ = sum(sum(wᵢ * xᵢ for (wᵢ, xᵢ) in zip(x, w)) for (x, w) in zip(xs, ws)) / n
    σ² = sum(sum(wᵢ * (xᵢ - μ)^2 for (wᵢ, xᵢ) in zip(x, w)) for (x, w) in zip(xs, ws)) / n
    return MyNormal(μ, σ²)
end
