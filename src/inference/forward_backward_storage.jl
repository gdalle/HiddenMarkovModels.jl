abstract type AbstractForwardBackwardStorage{R} end

## Normal

struct ForwardBackwardStorage{R} <: AbstractForwardBackwardStorage{R}
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
    logL = zero(R)
    for t in 1:duration(fb)
        logL -= log(fb.c[t])
    end
    return logL
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

## Semilog

struct SemiLogForwardBackwardStorage{R} <: AbstractForwardBackwardStorage{R}
    α::Matrix{R}
    β::Matrix{R}
    c::Vector{R}
    γ::Matrix{R}
    ξ::Array{R,3}
    _m::Vector{R}
    _Bβ::Matrix{R}
end

Base.length(fb::SemiLogForwardBackwardStorage) = size(fb.α, 1)
duration(fb::SemiLogForwardBackwardStorage) = size(fb.α, 2)

function loglikelihood(fb::SemiLogForwardBackwardStorage{R}) where {R}
    logL = zero(R)
    for t in 1:duration(fb)
        logL += -log(fb.c[t]) + fb._m[t]
    end
    return logL
end

function initialize_forward_backward(sp::StateProcess, logB, ::SemiLogScale)
    N, T = size(logB)
    p = initial_distribution(sp)
    A = transition_matrix(sp)
    R = promote_type(eltype(p), eltype(A), eltype(logB))
    α = Matrix{R}(undef, N, T)
    β = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    γ = Matrix{R}(undef, N, T)
    ξ = Array{R,3}(undef, N, N, T - 1)
    _m = Vector{R}(undef, T)
    _Bβ = Matrix{R}(undef, N, T)
    return SemiLogForwardBackwardStorage(α, β, c, γ, ξ, _m, _Bβ)
end

## Log

struct LogForwardBackwardStorage{R} <: AbstractForwardBackwardStorage{R}
    logα::Matrix{R}
    logβ::Matrix{R}
    logγ::Matrix{R}
    logξ::Array{R,3}
    _logαA::Matrix{R}  # not temporal
    _logABβ::Matrix{R}  # not temporal
end

Base.length(fb::LogForwardBackwardStorage) = size(fb.logα, 1)
duration(fb::LogForwardBackwardStorage) = size(fb.logα, 2)

function loglikelihood(fb::LogForwardBackwardStorage{R}) where {R}
    @views logL = logsumexp(fb.logα[:, end])
    return logL
end

function initialize_forward_backward(sp::StateProcess, logB, ::LogScale)
    N, T = size(logB)
    logp = log_initial_distribution(sp)
    logA = log_transition_matrix(sp)
    R = promote_type(eltype(logp), eltype(logA), eltype(logB))
    logα = Matrix{R}(undef, N, T)
    logβ = Matrix{R}(undef, N, T)
    logγ = Matrix{R}(undef, N, T)
    logξ = Array{R,3}(undef, N, N, T - 1)
    _logαA = Matrix{R}(undef, N, N)
    _logABβ = Matrix{R}(undef, N, N)
    return LogForwardBackwardStorage(logα, logβ, logγ, logξ, _logαA, _logABβ)
end
