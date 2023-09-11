"""
    ForwardBackwardStorage{R,M<:AbstractMatrix{R}}

Store forward-backward quantities with element type `R`.

# Fields

Let `X` denote the vector of hidden states and `Y` denote the vector of observations. The following fields are part of the API:

- `γ::Vector{Vector{R}}`: posterior one-state marginals `γ[i,t] = ℙ(X[t]=i | Y[1:T])`
- `ξ::Vector{M}`: posterior two-state marginals `ξ[t][i,j] = ℙ(X[t:t+1]=(i,j) | Y[1:T])`

The following fields are internals and subject to change:

- `α`: scaled forward variables `α[t][i]` proportional to `ℙ(Y[1:t], X[t]=i)` (up to a function of `t`)
- `β`: scaled backward variables `β[t][i]` proportional to `ℙ(Y[t+1:T] | X[t]=i)` (up to a function of `t`)
- `c`: forward variable inverse normalizations `c[t] = 1 / sum(α[:, t])`
- `logm`: maximum of the observation loglikelihoods `logB`
- `Bscaled`: numerically stabilized observation likelihoods `B`
- `Bβscaled`: numerically stabilized product `Bβ`
"""
struct ForwardBackwardStorage{R,M<:AbstractMatrix{R}}
    α::Vector{Vector{R}}
    β::Vector{Vector{R}}
    γ::Vector{Vector{R}}
    ξ::Vector{M}
    c::Vector{R}
    logm::Vector{R}
    Bscaled::Vector{Vector{R}}
    Bβscaled::Vector{Vector{R}}
end

Base.eltype(fb::ForwardBackwardStorage{R}) where {R} = R
Base.length(fb::ForwardBackwardStorage) = length(first(fb.α))
duration(fb::ForwardBackwardStorage) = length(fb.α)

function loglikelihood(fb::ForwardBackwardStorage{R}) where {R}
    logL = -sum(log, fb.c) + sum(fb.logm)
    return logL
end

function loglikelihood(fbs::Vector{ForwardBackwardStorage{R,M}}) where {R,M}
    logL = zero(R)
    for fb in fbs
        logL += loglikelihood(fb)
    end
    return logL
end

function initialize_forward_backward(p, A, logB)
    N, T = size(logB)
    R = promote_type(eltype(p), eltype(A), eltype(logB))
    V = Vector{R}
    M = typeof(similar(A, R))
    α = V[Vector{R}(undef, N) for t in 1:T]
    β = V[Vector{R}(undef, N) for t in 1:T]
    γ = V[Vector{R}(undef, N) for t in 1:T]
    ξ = M[similar(A, R) for t in 1:(T - 1)]
    c = Vector{R}(undef, T)
    logm = Vector{R}(undef, T)
    Bscaled = V[Vector{R}(undef, N) for t in 1:T]
    Bβscaled = V[Vector{R}(undef, N) for t in 1:T]
    return ForwardBackwardStorage(α, β, γ, ξ, c, logm, Bscaled, Bβscaled)
end

function initialize_forward_backward(hmm::AbstractHMM, logB)
    p = initial_distribution(hmm)
    A = transition_matrix(hmm)
    return initialize_forward_backward(p, A, logB)
end

function forward!(fb::ForwardBackwardStorage, p, A, logB)
    @unpack α, c, logm, Bscaled = fb
    T = length(α)
    maximum!(logm', logB)
    Bscaled[1] .= exp.(view(logB, :, 1) .- logm[1])
    α[1] .= p .* Bscaled[1]
    c[1] = inv(sum(α[1]))
    α[1] .*= c[1]
    for t in 1:(T - 1)
        Bscaled[t + 1] .= exp.(view(logB, :, t + 1) .- logm[t + 1])
        mul!(α[t + 1], A', α[t])
        α[t + 1] .*= Bscaled[t + 1]
        c[t + 1] = inv(sum(α[t + 1]))
        α[t + 1] .*= c[t + 1]
    end
    return nothing
end

function backward!(fb::ForwardBackwardStorage{R}, A, logB) where {R}
    @unpack β, c, Bscaled, Bβscaled = fb
    T = length(β)
    β[T] .= c[T]
    for t in (T - 1):-1:1
        Bβscaled[t + 1] .= Bscaled[t + 1] .* β[t + 1]
        mul!(β[t], A, Bβscaled[t + 1])
        β[t] .*= c[t]
    end
    Bβscaled[1] .= Bscaled[1] .* β[1]
    return nothing
end

function marginals!(fb::ForwardBackwardStorage, A)
    @unpack α, β, c, Bβscaled, γ, ξ = fb
    T = length(γ)
    for t in 1:T
        γ[t] .= α[t] .* β[t] ./ c[t]
    end
    for t in 1:(T - 1)
        ξ[t] .= A
        mul_rows!(ξ[t], α[t])
        mul_cols!(ξ[t], Bβscaled[t + 1])
    end
    return nothing
end

function forward_backward!(fb::ForwardBackwardStorage, p, A, logB)
    forward!(fb, p, A, logB)
    backward!(fb, A, logB)
    marginals!(fb, A)
    return nothing
end

function forward_backward!(fb::ForwardBackwardStorage, hmm::AbstractHMM, logB)
    p = initial_distribution(hmm)
    A = transition_matrix(hmm)
    return forward_backward!(fb, p, A, logB)
end

function forward_backward(p, A, logB)
    fb = initialize_forward_backward(p, A, logB)
    forward_backward!(fb, p, A, logB)
    return fb
end

function forward_backward(hmm::AbstractHMM, logB::Matrix)
    p = initial_distribution(hmm)
    A = transition_matrix(hmm)
    return forward_backward(p, A, logB)
end

"""
    forward_backward(hmm, obs_seq)

Apply the forward-backward algorithm to estimate the posterior state marginals of an HMM.

Return a [`ForwardBackwardStorage`](@ref).
"""
function forward_backward(hmm::AbstractHMM, obs_seq)
    logB = loglikelihoods(hmm, obs_seq)
    return forward_backward(hmm, logB)
end

"""
    forward_backward(hmm, obs_seqs, nb_seqs)

Apply the forward-backward algorithm to estimate the posterior state marginals of an HMM, based on multiple observation sequences.

Return a vector of [`ForwardBackwardStorage`](@ref) objects.

!!! warning "Multithreading"
    This function is parallelized across sequences.
"""
function forward_backward(hmm::AbstractHMM, obs_seqs, nb_seqs::Integer)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    fb1 = forward_backward(hmm, first(obs_seqs))
    fbs = Vector{typeof(fb1)}(undef, nb_seqs)
    fbs[1] = fb1
    @threads for k in 2:nb_seqs
        fbs[k] = forward_backward(hmm, obs_seqs[k])
    end
    return fbs
end
