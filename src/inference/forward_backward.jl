"""
$(TYPEDEF)

Store forward-backward quantities with element type `R`.

This storage is relative to a single sequence.

# Fields

The only fields useful outside of the algorithm are `γ`, `ξ`, the rest does not belong to the public API.

$(TYPEDFIELDS)
"""
struct ForwardBackwardStorage{R,M<:AbstractMatrix{R}}
    logL::RefValue{R}
    "scaled forward messsages `α[i,t]` proportional to `ℙ(Y[1:t], X[t]=i)` (up to a function of `t`)"
    α::Matrix{R}
    "scaled backward messsages `β[i,t]` proportional to `ℙ(Y[t+1:T] | X[t]=i)` (up to a function of `t`)"
    β::Matrix{R}
    "forward message inverse normalizations `c[t] = 1 / sum(α[:,t])`"
    c::Vector{R}
    "posterior state marginals `γ[i,t] = ℙ(X[t]=i | Y[1:T])`"
    γ::Matrix{R}
    "posterior transition marginals `ξ[t][i,j] = ℙ(X[t]=i, X[t+1]=j | Y[1:T])`"
    ξ::Vector{M}
    "observation loglikelihoods `logB[i,t] = ℙ(Y[t] | X[t]=i)`"
    logB::Matrix{R}
    "maximum of the observation loglikelihoods `logm[t] = maximum(logB[:, t])`"
    logm::Vector{R}
    "numerically stabilized observation likelihoods `B[i,t] = exp(logB[i,t] - logm[t])`"
    B::Matrix{R}
    "scratch storage space"
    scratch::Vector{R}
end

"""
    initialize_forward_backward(hmm, obs_seq)
    initialize_forward_backward(hmm, MultiSeq(obs_seqs))
"""
function initialize_forward_backward(
    hmm::AbstractHMM, obs_seq::Vector; transition_marginals=true
)
    N, T = length(hmm), length(obs_seq)
    trans = transition_matrix(hmm, 1)
    R = eltype(hmm, obs_seq[1])
    M = typeof(similar(trans, R))

    logL = RefValue{R}()
    α = Matrix{R}(undef, N, T)
    β = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    γ = Matrix{R}(undef, N, T)
    ξ = Vector{M}(undef, T - 1)
    if transition_marginals
        for t in 1:(T - 1)
            ξ[t] = similar(transition_matrix(hmm, t), R)
        end
    end
    logB = Matrix{R}(undef, N, T)
    logm = Vector{R}(undef, T)
    B = Matrix{R}(undef, N, T)
    scratch = Vector{R}(undef, N)
    return ForwardBackwardStorage{R,M}(logL, α, β, c, γ, ξ, logB, logm, B, scratch)
end

function initialize_forward_backward(
    hmm::AbstractHMM, obs_seqs::MultiSeq; transition_marginals=true
)
    R = eltype(hmm, obs_seqs[1][1])
    M = typeof(similar(transition_matrix(hmm, 1), R))
    storages = Vector{ForwardBackwardStorage{R,M}}(undef, length(obs_seqs))
    for k in eachindex(storages, obs_seqs)
        storages[k] = initialize_forward_backward(hmm, obs_seqs[k]; transition_marginals)
    end
    return storages
end

"""
    forward_backward!(storage, hmm, obs_seq)
    forward_backward!(storage, hmm, MultiSeq(obs_seqs))
"""
function forward_backward!(
    storage::ForwardBackwardStorage,
    hmm::AbstractHMM,
    obs_seq::Vector;
    transition_marginals::Bool=true,
)
    T = length(obs_seq)
    @unpack logL, α, β, c, γ, ξ, logB, logm, B, scratch = storage

    # Observation loglikelihoods then likelihoods
    @views for t in 1:T
        obs_logdensities!(logB[:, t], hmm, t, obs_seq[t])
    end
    maximum!(logm', logB)
    B .= exp.(logB .- logm')

    # Forward
    @views begin
        init = initialization(hmm)
        α[:, 1] .= init .* B[:, 1]
        c[1] = inv(sum(α[:, 1]))
        lmul!(c[1], α[:, 1])
    end
    @views for t in 1:(T - 1)
        trans = transition_matrix(hmm, t)
        mul!(α[:, t + 1], trans', α[:, t])
        α[:, t + 1] .*= B[:, t + 1]
        c[t + 1] = inv(sum(α[:, t + 1]))
        lmul!(c[t + 1], α[:, t + 1])
    end

    # Backward and transition marginals
    β[:, T] .= c[T]
    @views for t in (T - 1):-1:1
        trans = transition_matrix(hmm, t)
        scratch .= B[:, t + 1] .* β[:, t + 1]  # Bβ
        mul!(β[:, t], trans, scratch)
        lmul!(c[t], β[:, t])
        if transition_marginals
            # transition marginals using Bβ
            mul_rows_cols!(ξ[t], view(α, :, t), trans, scratch)
        end
    end

    # State marginals
    γ .= α .* β ./ c'
    check_finite(γ)

    # Loglikelihood
    logL[] = -sum(log, c) + sum(logm)
    return nothing
end

function forward_backward!(
    storages::Vector{<:ForwardBackwardStorage},
    hmm::AbstractHMM,
    obs_seqs::MultiSeq;
    transition_marginals::Bool=true,
)
    for k in eachindex(storages, obs_seqs)
        forward_backward!(storages[k], hmm, obs_seqs[k]; transition_marginals)
    end
end

"""
    forward_backward(hmm, obs_seq)

Run the forward-backward algorithm to infer the posterior state and transition marginals of `hmm` on the sequence `obs_seq`.

This function returns a tuple `(γ, logL)` where

- `γ` is a matrix containing the posterior state marginals `γ[i,t]` 
- `logL` is the loglikelihood of the sequence

# See also

- [`ForwardBackwardStorage`](@ref)
"""
function forward_backward(hmm::AbstractHMM, obs_seqs::MultiSeq)
    storages = initialize_forward_backward(hmm, obs_seqs; transition_marginals=false)
    forward_backward!(storages, hmm, obs_seqs; transition_marginals=false)
    return [(γ=storages[k].γ, logL=storages[k].logL[]) for k in eachindex(storages)]
end

function forward_backward(hmm::AbstractHMM, obs_seq::Vector)
    return only(forward_backward(hmm, MultiSeq([obs_seq])))
end
