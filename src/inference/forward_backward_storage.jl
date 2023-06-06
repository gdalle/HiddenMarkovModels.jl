"""
    ForwardBackwardStorage{R}

# Fields

- `α::Matrix{R}`: forward variables `α[i, t]`
- `c::Vector{R}`: forward variable inverse normalizations `c[t]`
- `β::Matrix{R}`: backward variables `β[i, t]`
- `Bβ::Matrix{R}`: auxiliary backward variables multiplied by the likelihoods `Bβ[i, t] = B[i, t] * β[i, t]`
- `γ::Matrix{R}`: one-state marginals `γ[i, t]`
- `ξ::Array{R,3}`: two-state marginals `ξ[i, j, t]`
"""
struct ForwardBackwardStorage{R}
    α::Matrix{R}
    c::Vector{R}
    β::Matrix{R}
    Bβ::Matrix{R}
    γ::Matrix{R}
    ξ::Array{R,3}
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

function forward_backward!(fb::ForwardBackwardStorage, p, A, B)
    (; α, c, β, Bβ, γ, ξ) = fb
    return forward_backward!(α, c, β, Bβ, γ, ξ, p, A, B)
end

"""
    forward_backward(hmm, obs_seq)

Apply the forward-backward algorithm to estimate the posterior state marginals of an HMM, and return a `ForwardBackwardStorage` object.
"""
function forward_backward(hmm::HMM, obs_seq)
    p = initial_distribution(hmm.state_process)
    A = transition_matrix(hmm.state_process)
    B = likelihoods(hmm.obs_process, obs_seq)
    fb = initialize_forward_backward(p, A, B)
    forward_backward!(fb, p, A, B)
    return fb
end

const MultiForwardBackwardStorage{R} = Vector{ForwardBackwardStorage{R}}

function loglikelihood(fbs::MultiForwardBackwardStorage{R}) where {R}
    return sum(loglikelihood, fbs)
end
