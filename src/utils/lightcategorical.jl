"""
$(TYPEDEF)

An HMMs-compatible implementation of a discrete categorical distribution, enabling allocation-free in-place estimation.

This is not part of the public API and is expected to change.

# Fields

$(TYPEDFIELDS)
"""
struct LightCategorical{T1,T2,V1<:AbstractVector{T1},V2<:AbstractVector{T2}}
    "vector of class probabilities"
    p::V1
    "vector of log class probabilities"
    logp::V2
end

function LightCategorical(p)
    check_prob_vec(p)
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
    for k in eachindex(dist.p)
        s += dist.p[k]
        if u < s
            return k
        end
    end
end

function DensityInterface.logdensityof(dist::LightCategorical, k::Integer)
    return dist.logp[k]
end

function StatsAPI.fit!(dist::LightCategorical{T1}, x, w) where {T1}
    w_tot = sum(w)
    dist.p .= zero(T1)
    for (xᵢ, wᵢ) in zip(x, w)
        dist.p[xᵢ] .+= wᵢ
    end
    dist.p ./= w_tot
    dist.logp .= log.(dist.p)
    check_prob_vec(dist.p)
    return nothing
end