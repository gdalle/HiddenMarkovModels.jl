"""
    ForwardBackwardStorage{R}

# Fields

- `α::Matrix{R}`: forward variables `α[i, t]`
- `β::Matrix{R}`: backward variables `β[i, t]`
- `c::Vector{R}`: forward variable inverse normalizations `c[t]`
- `γ::Matrix{R}`: one-state marginals `γ[i, t]`
- `ξ::Array{R,3}`: two-state marginals `ξ[i, j, t]`
"""
struct ForwardBackwardStorage{R}
    α::Matrix{R}
    β::Matrix{R}
    c::Vector{R}
    γ::Matrix{R}
    ξ::Array{R,3}
    _Bβ::Matrix{R}
end

Base.length(fb::ForwardBackwardStorage) = size(fb.α, 1)
duration(fb::ForwardBackwardStorage) = size(fb.α, 2)

function loglikelihood(fb::ForwardBackwardStorage{R}) where {R}
    LR = typeof(log(one(R)))
    logL = zero(LR)
    for t in 1:duration(fb)
        logL -= log(fb.c[t])
    end
    return logL
end

function loglikelihood(fbs::Vector{ForwardBackwardStorage{R}}) where {R}
    LR = typeof(log(one(R)))
    logL = zero(LR)
    for fb in fbs
        logL += loglikelihood(fb)
    end
    return logL
end

function forward!(fb::ForwardBackwardStorage, p, A, B)
    (; α, c) = fb
    T = size(α, 2)
    @views α[:, 1] .= p .* B[:, 1]
    @views c[1] = inv(sum(α[:, 1]))
    @views α[:, 1] .*= c[1]
    @views for t in 1:(T - 1)
        mul!(α[:, t + 1], A', α[:, t])
        α[:, t + 1] .*= B[:, t + 1]
        c[t + 1] = inv(sum(α[:, t + 1]))
        α[:, t + 1] .*= c[t + 1]
    end
    return nothing
end

function backward!(fb::ForwardBackwardStorage{R}, A, B) where {R}
    (; c, β, _Bβ) = fb
    T = size(β, 2)
    β[:, T] .= one(R)
    @views for t in (T - 1):-1:1
        _Bβ[:, t + 1] .= B[:, t + 1] .* β[:, t + 1]
        mul!(β[:, t], A, _Bβ[:, t + 1])
        β[:, t] .*= c[t]
    end
    return nothing
end

function marginals!(fb::ForwardBackwardStorage, A)
    (; α, β, _Bβ, γ, ξ) = fb
    T = size(γ, 2)
    @views for t in 1:T
        γ[:, t] .= α[:, t] .* β[:, t]
        normalization = inv(sum(γ[:, t]))
        γ[:, t] .*= normalization
    end
    @views for t in 1:(T - 1)
        ξ[:, :, t] .= α[:, t] .* A .* _Bβ[:, t + 1]'
        normalization = inv(sum(ξ[:, :, t]))
        ξ[:, :, t] .*= normalization
    end
    return nothing
end

function forward_backward!(fb::ForwardBackwardStorage, sp::StateProcess, B)
    p = initial_distribution(sp)
    A = transition_matrix(sp)
    forward!(fb, p, A, B)
    backward!(fb, A, B)
    marginals!(fb, A)
    return nothing
end

function initialize_forward_backward(sp::StateProcess, B, ::NormalScale)
    N, T = size(B)
    p = initial_distribution(sp)
    A = transition_matrix(sp)
    R = promote_type(eltype(p), eltype(A), eltype(B))
    α = Matrix{R}(undef, N, T)
    β = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    γ = Matrix{R}(undef, N, T)
    ξ = Array{R,3}(undef, N, N, T - 1)
    _Bβ = Matrix{R}(undef, N, T)
    return ForwardBackwardStorage(α, β, c, γ, ξ, _Bβ)
end
