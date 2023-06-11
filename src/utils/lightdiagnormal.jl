"""
    LightDiagonalNormal

An HMMs-compatible implementation of a multivariate normal distribution with diagonal covariance, enabling allocation-free estimation.

This is not part of the public API and is expected to change.
"""
struct LightDiagNormal{T1,T2,V1<:AbstractVector{T1},V2<:AbstractVector{T2}}
    μ::V1
    σ::V2
    logσ::V2
end

LightDiagNormal(μ, σ) = LightDiagNormal(μ, σ, log.(σ))

@inline DensityInterface.DensityKind(::LightDiagNormal) = HasDensity()

Base.length(dist::LightDiagNormal) = length(dist.μ)

function Base.rand(rng::AbstractRNG, dist::LightDiagNormal)
    return dist.σ .* randn(rng, length(dist)) .+ dist.μ
end

function DensityInterface.logdensityof(dist::LightDiagNormal, x)
    a = -sum(abs2, (x[i] - dist.μ[i]) / dist.σ[i] for i in eachindex(x, dist.μ, dist.σ))
    b = -sum(dist.logσ)
    return (a / 2) + b
end

function StatsAPI.fit!(dist::LightDiagNormal{T1,T2}, x, w) where {T1,T2}
    w_tot = sum(w)
    dist.μ .= zero(T1)
    dist.σ .= zero(T2)
    for i in eachindex(x, w)
        dist.μ .+= x[i] .* w[i]
    end
    dist.μ ./= w_tot
    for i in eachindex(x, w)
        dist.σ .+= abs2.(x[i] .- dist.μ) .* w[i]
    end
    dist.σ ./= w_tot
    dist.σ .= sqrt.(dist.σ)
    dist.logσ .= log.(dist.σ)
    return nothing
end
