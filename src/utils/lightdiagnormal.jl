"""
$(TYPEDEF)

An HMMs-compatible implementation of a multivariate normal distribution with diagonal covariance, enabling allocation-free in-place estimation.

This is not part of the public API and is expected to change.

# Fields

$(TYPEDFIELDS)
"""
struct LightDiagNormal{
    T1,T2,T3,V1<:AbstractVector{T1},V2<:AbstractVector{T2},V3<:AbstractVector{T3}
}
    "means"
    μ::V1
    "standard deviations"
    σ::V2
    "log standard deviations"
    logσ::V3
end

function LightDiagNormal(μ, σ)
    check_positive(σ)
    return LightDiagNormal(μ, σ, log.(σ))
end

function Base.show(io::IO, dist::LightDiagNormal)
    return print(io, "LightDiagNormal($(dist.μ), $(dist.σ))")
end

@inline DensityInterface.DensityKind(::LightDiagNormal) = HasDensity()

Base.length(dist::LightDiagNormal) = length(dist.μ)

function Base.rand(rng::AbstractRNG, dist::LightDiagNormal{T1,T2}) where {T1,T2}
    T = promote_type(T1, T2)
    return dist.σ .* randn(rng, T, length(dist)) .+ dist.μ
end

function DensityInterface.logdensityof(dist::LightDiagNormal, x)
    a = -sum(abs2, (x[i] - dist.μ[i]) / dist.σ[i] for i in eachindex(x, dist.μ, dist.σ))
    b = -sum(dist.logσ)
    logd = (a / 2) + b
    return logd
end

function StatsAPI.fit!(dist::LightDiagNormal{T1,T2}, x, w) where {T1,T2}
    w_tot = sum(w)
    dist.μ .= zero(T1)
    dist.σ .= zero(T2)
    for (xᵢ, wᵢ) in zip(x, w)
        dist.μ .+= xᵢ .* wᵢ
        dist.σ .+= abs2.(xᵢ) .* wᵢ
    end
    dist.μ ./= w_tot
    dist.σ ./= w_tot
    dist.σ .-= abs2.(dist.μ)
    dist.σ .= sqrt.(dist.σ)
    dist.logσ .= log.(dist.σ)
    check_positive(dist.σ)
    return nothing
end
