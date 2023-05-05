
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
    w_tot = sum(sum, ws)
    μ = sum(dot(w, x) for (x, w) in zip(xs, ws)) / w_tot
    σ² = zero(μ)
    for (x, w) in zip(xs, ws)
        for (xᵢ, wᵢ) in zip(x, w)
            σ² += wᵢ * (xᵢ - μ)^2
        end
    end
    σ² /= w_tot
    return MyNormal(μ, σ²)
end
