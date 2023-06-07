struct MyNormal{T1,T2}
    μ::T1
    σ²::T2
end

@inline DensityInterface.DensityKind(::MyNormal) = HasDensity()

function Base.rand(rng::AbstractRNG, dist::MyNormal)
    return sqrt(dist.σ²) * randn(rng) + dist.μ
end

function DensityInterface.densityof(dist::MyNormal, x)
    return exp(-abs2(x - dist.μ) / (2 * dist.σ²)) / sqrt(2π * dist.σ²)
end

function StatsAPI.fit(MN::Type{<:MyNormal}, xs, ws)
    w_tot = sum(ws)
    μ = dot(ws, xs) / w_tot
    σ² = zero(μ)
    for (x, w) in zip(xs, ws)
        σ² += w * (x - μ)^2
    end
    σ² /= w_tot
    return MN(μ, σ²)
end
