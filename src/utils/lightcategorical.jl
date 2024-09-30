"""
$(TYPEDEF)

An HMMs-compatible implementation of a discrete categorical distribution, enabling allocation-free in-place estimation.

This is not part of the public API and is expected to change.

# Fields

$(TYPEDFIELDS)
"""
struct LightCategorical{T1,T2,V1<:AbstractVector{T1},V2<:AbstractVector{T2}}
    "class probabilities"
    p::V1
    "log class probabilities"
    logp::V2
end

function LightCategorical(p::AbstractVector{T}) where {T}
    @argcheck valid_prob_vec(p)
    return LightCategorical(p, log.(p))
end

function Base.show(io::IO, dist::LightCategorical)
    return print(io, "LightCategorical($(dist.p))")
end

@inline DensityInterface.DensityKind(::LightCategorical) = HasDensity()

Base.length(dist::LightCategorical) = length(dist.p)

function Base.rand(rng::AbstractRNG, dist::LightCategorical{T1}) where {T1}
    u = rand(rng)
    s = zero(T1)
    x = 0
    for k in eachindex(dist.p)
        s += dist.p[k]
        if u <= s
            x = k
            return x
        end
    end
    return x
end

function DensityInterface.logdensityof(dist::LightCategorical, k::Integer)
    return dist.logp[k]
end

function StatsAPI.fit!(
    dist::LightCategorical{T1}, x::AbstractVector{<:Integer}, w::AbstractVector
) where {T1}
    @argcheck 1 <= minimum(x) <= maximum(x) <= length(dist.p)
    w_tot = sum(w)
    fill!(dist.p, zero(T1))
    @simd for i in eachindex(x, w)
        dist.p[x[i]] += w[i]
    end
    dist.p ./= w_tot
    dist.logp .= log.(dist.p)
    @argcheck valid_prob_vec(dist.p)
    return nothing
end
