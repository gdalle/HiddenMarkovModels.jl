"""
$(TYPEDEF)

Store forward-backward quantities with element type `R`.

This storage is relative to a single sequence.

# Fields

Let `X` denote the vector of hidden states and `Y` denote the vector of observations.

$(TYPEDFIELDS)
"""
struct ForwardBackwardStorage{
    R,V<:AbstractVector{R},M<:AbstractMatrix{R},A3<:AbstractArray{R,3}
}
    "total loglikelihood"
    logL::RefValue{R}
    "scaled forward variables `α[i,t]` proportional to `ℙ(Y[1:t], X[t]=i)` (up to a function of `t`)"
    α::M
    "scaled backward variables `β[i,t]` proportional to `ℙ(Y[t+1:T] | X[t]=i)` (up to a function of `t`)"
    β::M
    "posterior state marginals `γ[i,t] = ℙ(X[t]=i | Y[1:T])`"
    γ::M
    "posterior transition marginals `ξ[i,j,t] = ℙ(X[t:t+1]=(i,j) | Y[1:T])`"
    ξ::A3
    "forward variable inverse normalizations `c[t] = 1 / sum(α[:,t])`"
    c::V
    "observation loglikelihoods `logB[i, t]`"
    logB::M
    "maximum of the observation loglikelihoods `logm[t] = maximum(logB[:, t])`"
    logm::V
    "numerically stabilized observation likelihoods `B̃[i,t] = exp.(logB[i,t] - logm[t])`"
    B̃::M
    "numerically stabilized product `B̃β[i,t] = B̃[i,t] * β[i,t]`"
    B̃β::M
end

Base.eltype(::ForwardBackwardStorage{R}) where {R} = R
Base.length(fb::ForwardBackwardStorage) = size(fb.α, 1)
duration(fb::ForwardBackwardStorage) = size(fb.α, 2)

function Base.view(fb::ForwardBackwardStorage{R}, r::AbstractUnitRange) where {R}
    logL = Ref(zero(R))
    α = view(fb.α, :, r)
    β = view(fb.β, :, r)
    γ = view(fb.γ, :, r)
    ξ = view(fb.ξ, :, :, r)
    c = view(fb.c, r)
    logB = view(fb.logB, :, r)
    logm = view(fb.logm, r)
    B̃ = view(fb.B̃, :, r)
    B̃β = view(fb.B̃β, :, r)
    return ForwardBackwardStorage(logL, α, β, γ, ξ, c, logB, logm, B̃, B̃β)
end

function initialize_forward_backward(hmm::AbstractHMM, obs_seq::Vector)
    N, T = length(hmm), length(obs_seq)
    R = eltype(hmm, obs_seq[1])

    logL = RefValue{R}(zero(R))
    α = Matrix{R}(undef, N, T)
    β = Matrix{R}(undef, N, T)
    γ = Matrix{R}(undef, N, T)
    ξ = Array{R,3}(undef, N, N, T)
    c = Vector{R}(undef, T)
    logB = Matrix{R}(undef, N, T)
    logm = Vector{R}(undef, T)
    B̃ = Matrix{R}(undef, N, T)
    B̃β = Matrix{R}(undef, N, T)

    return ForwardBackwardStorage(logL, α, β, γ, ξ, c, logB, logm, B̃, B̃β)
end

function update_likelihoods!(fb::ForwardBackwardStorage, hmm::AbstractHMM, obs_seq::Vector)
    d = obs_distributions(hmm)
    @unpack logB, logm, B̃ = fb

    for t in eachindex(axes(logB, 2), obs_seq)
        logB[:, t] .= logdensityof.(d, (obs_seq[t],))
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
    fb.logL[] = -sum(log, fb.c) + sum(fb.logm)
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
    ξ[:, :, T] .= zero(eltype(ξ))
    check_no_nan(ξ)
    return nothing
end

function forward_backward!(fb::ForwardBackwardStorage, hmm::AbstractHMM, obs_seq::Vector)
    update_likelihoods!(fb, hmm, obs_seq)
    forward!(fb, hmm)
    backward!(fb, hmm)
    marginals!(fb, hmm)
    return nothing
end

function forward_backward!(
    fbs::Vector{<:ForwardBackwardStorage}, hmm::AbstractHMM, obs_seqs::Vector{<:Vector}
)
    @threads for k in eachindex(fbs, obs_seqs)
        forward_backward!(fbs[k], hmm, obs_seqs[k])
    end
    return nothing
end

"""
    forward_backward(hmm, obs_seq)
    forward_backward(hmm, obs_seqs, nb_seqs)

Run the forward-backward algorithm to infer the posterior state and transition marginals of an HMM.

When applied on a single sequence, this function returns a tuple `(γ, ξ, logL)` where

- `γ` is a matrix containing the posterior state marginals `γ[i, t]` 
- `ξ` is a 3-tensor containing the posterior transition marginals `ξ[i, j, t]`
- `logL` is the loglikelihood of the sequence

WHen applied on multiple sequences, it returns a vector of tuples.
"""
function forward_backward(hmm::AbstractHMM, obs_seq::Vector)
    fb = initialize_forward_backward(hmm, obs_seq)
    forward_backward!(fb, hmm, obs_seq)
    return (fb.γ, fb.ξ, fb.logL[])
end

function forward_backward(hmm::AbstractHMM, obs_seqs::Vector{<:Vector}, nb_seqs::Integer)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    fbs = [initialize_forward_backward(hmm, obs_seqs[k]) for k in eachindex(obs_seqs)]
    forward_backward!(fbs, hmm, obs_seqs)
    return [(fb.γ, fb.ξ, fb.logL[]) for fb in fbs]
end
