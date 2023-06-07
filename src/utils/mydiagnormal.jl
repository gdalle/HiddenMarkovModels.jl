struct MyDiagNormal{T1,T2}
    μ::Vector{T1}
    σ²::Vector{T2}
end

@inline DensityInterface.DensityKind(::MyDiagNormal) = HasDensity()

Base.length(dist::MyDiagNormal) = length(dist.μ)

function Base.rand(rng::AbstractRNG, dist::MyDiagNormal)
    return sqrt.(dist.σ²) .* randn(rng, length(dist)) .+ dist.μ
end

function DensityInterface.densityof(dist::MyDiagNormal, x)
    return prod(
        inv(sqrt(2π * σᵢ²)) * exp(-(xᵢ - μᵢ)^2 * inv(2 * σᵢ²)) for
        (xᵢ, μᵢ, σᵢ²) in zip(x, dist.μ, dist.σ²)
    )
end

function StatsAPI.fit(::Type{MyDiagNormal{T1,T2}}, xs, ws) where {T1,T2}
    n = length(first(xs))
    μ = zeros(T1, n)
    σ² = zeros(T2, n)
    w_tot = sum(ws)
    for (x, w) in zip(xs, ws)
        μ .+= w .* x
    end
    μ ./= w_tot
    for (x, w) in zip(xs, ws)
        σ² .+= w .* (x .- μ) .^ 2
    end
    σ² ./= w_tot
    return MyDiagNormal{T1,T2}(μ, σ²)
end
