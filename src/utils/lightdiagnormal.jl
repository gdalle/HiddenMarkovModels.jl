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

function LightDiagNormal(μ::AbstractVector{T1}, σ::AbstractVector{T2}) where {T1,T2}
    logσ = log.(σ)
    @argcheck all(isfinite, μ)
    @argcheck all(isfinite, σ)
    @argcheck all(isfinite, logσ)
    return LightDiagNormal(μ, σ, logσ)
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

function DensityInterface.logdensityof(
    dist::LightDiagNormal{T1,T2,T3}, x::AbstractVector
) where {T1,T2,T3}
    l = zero(promote_type(T1, T2, T3, eltype(x)))
    l -= sum(dist.logσ) + log2π * length(x) / 2
    @simd for i in eachindex(x, dist.μ, dist.σ)
        l -= abs2(x[i] - dist.μ[i]) / (2 * abs2(dist.σ[i]))
    end
    return l
end

function StatsAPI.fit!(
    dist::LightDiagNormal{T1,T2}, x::AbstractVector{<:AbstractVector}, w::AbstractVector
) where {T1,T2}
    w_tot = sum(w)
    fill!(dist.μ, zero(T1))
    fill!(dist.σ, zero(T2))
    @simd for i in eachindex(x, w)
        axpy!(w[i], x[i], dist.μ)
    end
    dist.μ ./= w_tot
    @simd for i in eachindex(x, w)
        dist.σ .+= abs2.(x[i] .- dist.μ) .* w[i]
    end
    dist.σ .= sqrt.(dist.σ ./ w_tot)
    dist.logσ .= log.(dist.σ)
    @argcheck all(isfinite, dist.μ)
    @argcheck all(isfinite, dist.σ)
    @argcheck all(isfinite, dist.logσ)
    return nothing
end
