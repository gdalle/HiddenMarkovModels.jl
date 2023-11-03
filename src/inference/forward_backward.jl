"""
$(TYPEDEF)

Store forward-backward quantities with element type `R`.

# Fields

Let `X` denote the vector of hidden states and `Y` denote the vector of observations.

$(TYPEDFIELDS)

Only the `γ` and `ξ` fields are part of the public API.
"""
struct ForwardBackwardStorage{R}
    "scaled forward variables `α[i,t]` proportional to `ℙ(Y[1:t], X[t]=i)` (up to a function of `t`)"
    α::Matrix{R}
    "scaled backward variables `β[i,t]` proportional to `ℙ(Y[t+1:T] | X[t]=i)` (up to a function of `t`)"
    β::Matrix{R}
    "posterior one-state marginals `γ[i,t] = ℙ(X[t]=i | Y[1:T])`"
    γ::Matrix{R}
    "posterior two-state marginals `ξ[i,j,t] = ℙ(X[t:t+1]=(i,j) | Y[1:T])`"
    ξ::Array{R,3}
    "forward variable inverse normalizations `c[t] = 1 / sum(α[:,t])`"
    c::Vector{R}
    "observation loglikelihoods `logB[i, t]`"
    logB::Matrix{R}
    "maximum of the observation loglikelihoods `logm[t] = maximum(logB[:, t])`"
    logm::Vector{R}
    "numerically stabilized observation likelihoods `B̃[i,t] = exp.(logB[i,t] - logm[t])`"
    B̃::Matrix{R}
    "numerically stabilized product `B̃β[i,t] = B̃[i,t] * β[i,t]`"
    B̃β::Matrix{R}
end

Base.eltype(::ForwardBackwardStorage{R}) where {R} = R
Base.length(fb::ForwardBackwardStorage) = size(fb.α, 1)
duration(fb::ForwardBackwardStorage) = size(fb.α, 2)

function loglikelihood(fb::ForwardBackwardStorage{R}) where {R}
    logL = -sum(log, fb.c) + sum(fb.logm)
    return logL
end

function loglikelihood(fbs::Vector{ForwardBackwardStorage{R}}) where {R}
    logL = zero(R)
    for fb in fbs
        logL += loglikelihood(fb)
    end
    return logL
end

function initialize_forward_backward(hmm::AbstractHMM, obs_seq)
    p = initialization(hmm)
    A = transition_matrix(hmm)
    testval = logdensityof(obs_distribution(hmm, 1), obs_seq[1])
    R = promote_type(eltype(p), eltype(A), typeof(testval))

    N, T = length(hmm), length(obs_seq)
    α = Matrix{R}(undef, N, T)
    β = Matrix{R}(undef, N, T)
    γ = Matrix{R}(undef, N, T)
    ξ = Array{R,3}(undef, N, N, T - 1)
    c = Vector{R}(undef, T)
    logB = Matrix{R}(undef, N, T)
    logm = Vector{R}(undef, T)
    B̃ = Matrix{R}(undef, N, T)
    B̃β = Matrix{R}(undef, N, T)

    return ForwardBackwardStorage(α, β, γ, ξ, c, logB, logm, B̃, B̃β)
end

function update_likelihoods!(fb::ForwardBackwardStorage, hmm::AbstractHMM, obs_seq)
    @unpack logB, logm, B̃ = fb
    d = obs_distributions(hmm)
    for (logb, obs) in zip(eachcol(logB), obs_seq)
        logb .= logdensityof(d, Ref(obs))
    end
    maximum!(logm', logB)
    B̃ .= exp.(logB .- logm')
    return nothing
end

function forward!(fb::ForwardBackwardStorage, hmm::AbstractHMM)
    p = initialization(hmm)
    A = transition_matrix(hmm)
    @unpack α, c, B̃ = fb
    T = size(α, 2)
    @views begin
        α[:, 1] .= p .* B̃[:, 1]
        c[1] = inv(sum(α[:, 1]))
        α[:, 1] .*= c[1]
    end
    @views for t in 1:(T - 1)
        mul!(α[:, t + 1], A', α[:, t])
        α[:, t + 1] .*= B̃[:, t + 1]
        c[t + 1] = inv(sum(α[:, t + 1]))
        α[:, t + 1] .*= c[t + 1]
    end
    check_no_nan(α)
    return nothing
end

function backward!(fb::ForwardBackwardStorage{R}, hmm::AbstractHMM) where {R}
    A = transition_matrix(hmm)
    @unpack β, c, B̃, B̃β = fb
    T = size(β, 2)
    β[:, T] .= c[T]
    @views for t in (T - 1):-1:1
        B̃β[:, t + 1] .= B̃[:, t + 1] .* β[:, t + 1]
        mul!(β[:, t], A, B̃β[:, t + 1])
        β[:, t] .*= c[t]
    end
    @views B̃β[:, 1] .= B̃[:, 1] .* β[:, 1]
    check_no_nan(β)
    return nothing
end

function marginals!(fb::ForwardBackwardStorage, hmm::AbstractHMM)
    A = transition_matrix(hmm)
    @unpack α, β, c, B̃β, γ, ξ = fb
    N, T = size(γ)
    γ .= α .* β ./ c'
    check_no_nan(γ)
    @views for t in 1:(T - 1)
        ξ[:, :, t] .= α[:, t] .* A .* B̃β[:, t + 1]'
    end
    check_no_nan(ξ)
    return nothing
end

function forward_backward!(fb::ForwardBackwardStorage, hmm::AbstractHMM, obs_seq)
    update_likelihoods!(fb, hmm, obs_seq)
    forward!(fb, hmm)
    backward!(fb, hmm)
    marginals!(fb, hmm)
    return nothing
end

"""
    forward_backward(hmm, obs_seq)

Apply the forward-backward algorithm to estimate the posterior state marginals of an HMM.

Return a [`ForwardBackwardStorage`](@ref).
"""
function forward_backward(hmm::AbstractHMM, obs_seq)
    fb = initialize_forward_backward(hmm, obs_seq)
    forward_backward!(fb, hmm, obs_seq)
    return fb
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
    @threads for k in eachindex(obs_seqs, fbs)
        if k > 2
            fbs[k] = forward_backward(hmm, obs_seqs[k])
        end
    end
    return fbs
end
