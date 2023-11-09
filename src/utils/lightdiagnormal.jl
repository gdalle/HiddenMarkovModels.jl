"""
$(TYPEDEF)

An HMMs-compatible implementation of a multivariate normal distribution with diagonal covariance, enabling allocation-free estimation.

This is not part of the public API and is expected to change.

# Fields

$(TYPEDFIELDS)
"""
struct LightDiagNormal{
    T1,T2,T3,V1<:AbstractVector{T1},V2<:AbstractVector{T2},V3<:AbstractVector{T3}
}
    "vector of means"
    μ::V1
    "vector of standard deviations"
    σ::V2
    "vector of log standard deviations"
    logσ::V3
end

function LightDiagNormal(μ, σ)
    check_no_nan(μ)
    check_positive(σ)
    return LightDiagNormal(μ, σ, log.(σ))
end

function Base.show(io::IO, dist::LightDiagNormal)
    return print(io, "LightDiagNormal($(dist.μ), $(dist.σ))")
end

@inline DensityInterface.DensityKind(::LightDiagNormal) = HasDensity()

Base.length(dist::LightDiagNormal) = length(dist.μ)

function Base.rand(rng::AbstractRNG, dist::LightDiagNormal)
    return dist.σ .* randn(rng, length(dist)) .+ dist.μ
end

function DensityInterface.logdensityof(dist::LightDiagNormal, x)
    a = -sum(abs2, (x[i] - dist.μ[i]) / dist.σ[i] for i in eachindex(x, dist.μ, dist.σ))
    b = -sum(dist.logσ)
    check_no_nan(a)
    check_no_nan(b)
    logd = (a / 2) + b
    return logd
end

function StatsAPI.fit!(dist::LightDiagNormal{T1,T2}, x, w) where {T1,T2}
    w_tot = sum(w)
    check_positive(w_tot)
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
    check_positive(dist.σ)
    dist.logσ .= log.(dist.σ)
    return nothing
end
