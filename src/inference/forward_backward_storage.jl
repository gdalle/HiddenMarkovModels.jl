struct ForwardBackwardStorage{R}
    α::Matrix{R}
    β::Matrix{R}
    γ::Matrix{R}
    ξ::Array{R,3}
    _c::Vector{R}
    _m::Vector{R}
    _Bβ::Matrix{R}
end

Base.length(fb::ForwardBackwardStorage) = size(fb.α, 1)
duration(fb::ForwardBackwardStorage) = size(fb.α, 2)

function loglikelihood(fb::ForwardBackwardStorage{R}) where {R}
    logL = zero(R)
    for t in 1:duration(fb)
        logL += -log(fb._c[t]) + fb._m[t]
    end
    return logL
end

function loglikelihood(fbs::Vector{ForwardBackwardStorage{R}}) where {R}
    logL = zero(R)
    for fb in fbs
        logL += loglikelihood(fb)
    end
    return logL
end

function initialize_forward_backward(sp::StateProcess, logB)
    N, T = size(logB)
    p = initial_distribution(sp)
    A = transition_matrix(sp)
    R = promote_type(eltype(p), eltype(A), eltype(logB))
    α = Matrix{R}(undef, N, T)
    β = Matrix{R}(undef, N, T)
    γ = Matrix{R}(undef, N, T)
    ξ = Array{R,3}(undef, N, N, T - 1)
    _c = Vector{R}(undef, T)
    _m = Vector{R}(undef, T)
    _Bβ = Matrix{R}(undef, N, T)
    return ForwardBackwardStorage(α, β, γ, ξ, _c, _m, _Bβ)
end
