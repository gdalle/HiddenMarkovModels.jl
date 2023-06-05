"""
    forward_backward!(α, c, β, Bβ, γ, ξ, p, A, B)

Apply the full forward-backward algorithm by mutating `α`, `c`, `β`, `Bβ`, `γ` and `ξ`.
"""
function forward_backward!(
    α::Matrix, c::Vector, β::Matrix, Bβ::Matrix, γ::Matrix, ξ::Array{<:Any,3}, p, A, B
)
    forward!(α, c, p, A, B)
    backward!(β, Bβ, c, A, B)
    marginals!(γ, ξ, α, β, Bβ, A)
    logL = -sum(log, c)
    return logL
end

Base.@kwdef struct ForwardBackwardStorage{R}
    α::Matrix{R}
    c::Vector{R}
    β::Matrix{R}
    Bβ::Matrix{R}
    γ::Matrix{R}
    ξ::Array{R,3}
end

const MultiForwardBackwardStorage{R} = Vector{ForwardBackwardStorage{R}}

nb_states(fb::ForwardBackwardStorage) = size(fb.α, 1)
duration(fb::ForwardBackwardStorage) = size(fb.α, 2)

function forward_backward!(fb::ForwardBackwardStorage, p, A, B)
    (; α, c, β, Bβ, γ, ξ) = fb
    return forward_backward!(α, c, β, Bβ, γ, ξ, p, A, B)
end

function initialize_forward_backward(p, A, B)
    N, T = size(B)
    R = promote_type(eltype(p), eltype(A), eltype(B))
    α = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    β = Matrix{R}(undef, N, T)
    Bβ = Matrix{R}(undef, N, T)
    γ = Matrix{R}(undef, N, T)
    ξ = Array{R,3}(undef, N, N, T - 1)
    return ForwardBackwardStorage(α, c, β, Bβ, γ, ξ)
end

function forward_backward(hmm::HMM, obs_seq::Vector)
    p = initial_distribution(hmm)
    A = transition_matrix(hmm)
    B = likelihoods(hmm, obs_seq)
    fb = initialize_forward_backward(p, A, B)
    logL = forward_backward!(fb, p, A, B)
    return fb, logL
end
