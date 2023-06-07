struct MyNormal{T1,T2}
    μ::T1
    logσ²::T2
end

@inline DensityInterface.DensityKind(::MyNormal) = HasDensity()

function Base.rand(rng::AbstractRNG, dist::MyNormal)
    return sqrt(exp(dist.logσ²)) * randn(rng) + dist.μ
end

function DensityInterface.logdensityof(dist::MyNormal, x)
    return -abs2(x - dist.μ) / (2 * exp(dist.logσ²)) / sqrt(2π * exp(dist.logσ²))
end

function StatsAPI.fit(::Type{MyNormal{T1,T2}}, xs, ws) where {T1,T2}
    w_tot = sum(ws)
    μ = zero(T1)
    for (x, w) in zip(xs, ws)
        μ += w * x
    end
    μ /= w_tot
    σ² = zero(T2)
    for (x, w) in zip(xs, ws)
        σ² += w * (x - μ)^2
    end
    σ² /= w_tot
    return MyNormal{T1,T2}(μ, log(σ²))
end
