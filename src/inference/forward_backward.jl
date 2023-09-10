"""
    ForwardBackwardStorage{R}

Store forward-backward quantities with element type `R`.

# Fields

Let `X` denote the vector of hidden states and `Y` denote the vector of observations. The following fields are part of the API:

- `γ::Matrix{R}`: posterior one-state marginals `γ[i,t] = ℙ(X[t]=i | Y[1:T])`
- `ξ::Array{R,3}`: posterior two-state marginals `ξ[i,j,t] = ℙ(X[t:t+1]=(i,j) | Y[1:T])`

The following fields are internals and subject to change:

- `α::Matrix{R}`: scaled forward variables `α[i,t]` proportional to `ℙ(Y[1:t], X[t]=i)` (up to a function of `t`)
- `β::Matrix{R}`: scaled backward variables `β[i,t]` proportional to `ℙ(Y[t+1:T] | X[t]=i)` (up to a function of `t`)
- `c::Vector{R}`: forward variable inverse normalizations `c[t] = 1 / sum(α[:, t])`
- `m::Vector{R}`: maximum of the observation loglikelihoods `logB`
- `Bscaled::Matrix{R}`: numerically stabilized observation likelihoods `B`
- `Bβscaled::Matrix{R}`: numerically stabilized product `Bβ`
"""
struct ForwardBackwardStorage{R}
    α::Matrix{R}
    β::Matrix{R}
    γ::Matrix{R}
    ξ::Array{R,3}
    c::Vector{R}
    m::Vector{R}
    Bscaled::Matrix{R}
    Bβscaled::Matrix{R}
end

Base.length(fb::ForwardBackwardStorage) = size(fb.α, 1)
duration(fb::ForwardBackwardStorage) = size(fb.α, 2)

function loglikelihood(fb::ForwardBackwardStorage{R}) where {R}
    logL = zero(R)
    for t in 1:duration(fb)
        logL += -log(fb.c[t]) + fb.m[t]
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

function initialize_forward_backward(p, A, logB)
    N, T = size(logB)
    R = promote_type(eltype(p), eltype(A), eltype(logB))
    α = Matrix{R}(undef, N, T)
    β = Matrix{R}(undef, N, T)
    γ = Matrix{R}(undef, N, T)
    ξ = Array{R,3}(undef, N, N, T - 1)
    c = Vector{R}(undef, T)
    m = Vector{R}(undef, T)
    Bscaled = Matrix{R}(undef, N, T)
    Bβscaled = Matrix{R}(undef, N, T)
    return ForwardBackwardStorage(α, β, γ, ξ, c, m, Bscaled, Bβscaled)
end

function initialize_forward_backward(hmm::AbstractHMM, logB)
    p = initial_distribution(hmm)
    A = transition_matrix(hmm)
    return initialize_forward_backward(p, A, logB)
end

function forward!(fb::ForwardBackwardStorage, p, A, logB)
    @unpack α, c, m, Bscaled = fb
    T = size(α, 2)
    maximum!(m', logB)
    Bscaled .= exp.(logB .- m')
    @views begin
        α[:, 1] .= p .* Bscaled[:, 1]
        c[1] = inv(sum(α[:, 1]))
        α[:, 1] .*= c[1]
    end
    @views for t in 1:(T - 1)
        mul!(α[:, t + 1], A', α[:, t])
        α[:, t + 1] .*= Bscaled[:, t + 1]
        c[t + 1] = inv(sum(α[:, t + 1]))
        α[:, t + 1] .*= c[t + 1]
    end
    check_no_nan(α)
    return nothing
end

function backward!(fb::ForwardBackwardStorage{R}, A, logB) where {R}
    @unpack β, c, Bscaled, Bβscaled = fb
    T = size(β, 2)
    β[:, T] .= c[T]
    @views for t in (T - 1):-1:1
        Bβscaled[:, t + 1] .= Bscaled[:, t + 1] .* β[:, t + 1]
        mul!(β[:, t], A, Bβscaled[:, t + 1])
        β[:, t] .*= c[t]
    end
    @views Bβscaled[:, 1] .= Bscaled[:, 1] .* β[:, 1]
    check_no_nan(β)
    return nothing
end

function marginals!(fb::ForwardBackwardStorage, A)
    @unpack α, β, c, Bβscaled, γ, ξ = fb
    N, T = size(γ)
    γ .= α .* β ./ c'
    check_no_nan(γ)
    ξ .= (
        reshape(α[:, 1:(T - 1)], N, 1, T - 1) .* #
        reshape(A, N, N, 1) .* #
        reshape(Bβscaled[:, 2:T], 1, N, T - 1)
    )
    check_no_nan(ξ)
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
