struct MyDiagNormal{T1,T2}
    μ::T1
    σ²::T2
    n::Int
end

@inline DensityInterface.DensityKind(::MyDiagNormal) = HasDensity()

function Base.rand(rng::AbstractRNG, dist::MyDiagNormal)
    return sqrt(dist.σ²) .* randn(rng, dist.n) .+ dist.μ
end

function DensityInterface.densityof(dist::MyDiagNormal, x)
    return prod(
        inv(sqrt(2π * dist.σ²)) * exp(-(xᵢ - dist.μ)^2 * inv(2 * dist.σ²)) for xᵢ in x
    )
end

function StatsAPI.fit(MDN::Type{<:MyDiagNormal}, xs, ws)
    n = length(first(xs))
    w_tot = sum(ws) * n
    μ = sum(w * sum(x) for (x, w) in zip(xs, ws)) / w_tot
    σ² = zero(μ)
    for (x, w) in zip(xs, ws)
        for xᵢ in x
            σ² += w * (xᵢ - μ)^2
        end
    end
    σ² /= w_tot
    return MDN(μ, σ², n)
end
