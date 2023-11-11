"""
$(TYPEDEF)

Store forward-backward quantities with element type `R`.

This storage is relative to a single sequence.

# Fields

The only fields useful outside of the algorithm are `γ`, `logL`, `init_count` and `trans_count`.

$(TYPEDFIELDS)
"""
struct ForwardBackwardStorage{R,M<:AbstractMatrix{R}}
    "total loglikelihood"
    logL::RefValue{R}
    "scaled forward messsages `α[i,t]` proportional to `ℙ(Y[1:t], X[t]=i)` (up to a function of `t`)"
    α::Matrix{R}
    "scaled backward messsages `β[i,t]` proportional to `ℙ(Y[t+1:T] | X[t]=i)` (up to a function of `t`)"
    β::Matrix{R}
    "posterior state marginals `γ[i,t] = ℙ(X[t]=i | Y[1:T])`"
    γ::Matrix{R}
    "forward message inverse normalizations `c[t] = 1 / sum(α[:,t])`"
    c::Vector{R}
    "observation loglikelihoods `logB[i,t] = ℙ(Y[t] | X[t]=i)`"
    logB::Matrix{R}
    "maximum of the observation loglikelihoods `logm[t] = maximum(logB[:, t])`"
    logm::Vector{R}
    "numerically stabilized observation likelihoods `B[i,t] = exp.(logB[i,t] - logm[t])`"
    B::Matrix{R}
    "product `Bβ[i,t] = B[i,t] * β[i,t]`"
    Bβ::Matrix{R}
    "posterior initialization count"
    init_count::Vector{R}
    "posterior transition count"
    trans_count::M
end

function initialize_forward_backward(hmm::AbstractHMM, obs_seq::Vector)
    N, T = length(hmm), length(obs_seq)
    A = transition_matrix(hmm)
    R = eltype(hmm, obs_seq[1])

    logL = RefValue{R}(zero(R))
    α = Matrix{R}(undef, N, T)
    β = Matrix{R}(undef, N, T)
    γ = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    logB = Matrix{R}(undef, N, T)
    logm = Vector{R}(undef, T)
    B = Matrix{R}(undef, N, T)
    Bβ = Matrix{R}(undef, N, T)
    init_count = Vector{R}(undef, N)
    trans_count = similar(A, R)
    M = typeof(trans_count)
    return ForwardBackwardStorage{R,M}(
        logL, α, β, γ, c, logB, logm, B, Bβ, init_count, trans_count
    )
end

function forward_backward!(fb::ForwardBackwardStorage, hmm::AbstractHMM, obs_seq::Vector)
    p = initialization(hmm)
    A = transition_matrix(hmm)
    T = length(obs_seq)
    @unpack α, β, c, γ, logB, logm, B, Bβ, init_count, trans_count = fb

    # Observation loglikelihoods
    for (logb, obs) in zip(eachcol(logB), obs_seq)
        obs_logdensities!(logb, hmm, obs)
    end
    check_right_finite(logB)
    maximum!(logm', logB)
    B .= exp.(logB .- logm')

    # Forward
    @views begin
        α[:, 1] .= p .* B[:, 1]
        c[1] = inv(sum(α[:, 1]))
        lmul!(c[1], α[:, 1])
    end
    @views for t in 1:(T - 1)
        mul!(α[:, t + 1], A', α[:, t])
        α[:, t + 1] .*= B[:, t + 1]
        c[t + 1] = inv(sum(α[:, t + 1]))
        lmul!(c[t + 1], α[:, t + 1])
    end

    # Backward
    β[:, T] .= c[T]
    @views for t in (T - 1):-1:1
        Bβ[:, t + 1] .= B[:, t + 1] .* β[:, t + 1]
        mul!(β[:, t], A, Bβ[:, t + 1])
        lmul!(c[t], β[:, t])
    end
    @views Bβ[:, 1] .= B[:, 1] .* β[:, 1]

    # Marginals
    γ .= α .* β ./ c'
    check_finite(γ)

    # Sufficient stats
    init_count .= @view γ[:, 1]
    trans_count .= zero(eltype(trans_count))
    @views for t in 1:(T - 1)
        add_mul_rows_cols!(trans_count, α[:, t], A, Bβ[:, t + 1])
    end

    # Loglikelihood
    fb.logL[] = -sum(log, fb.c) + sum(fb.logm)

    return nothing
end

function forward_backward!(
    fbs::Vector{<:ForwardBackwardStorage},
    hmm::AbstractHMM,
    obs_seqs::Vector{<:Vector},
    nb_seqs::Integer,
)
    check_nb_seqs(obs_seqs, nb_seqs)
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
- `logL` is the loglikelihood of the sequence

WHen applied on multiple sequences, it returns a vector of tuples.
"""
function forward_backward(hmm::AbstractHMM, obs_seqs::Vector{<:Vector}, nb_seqs::Integer)
    check_nb_seqs(obs_seqs, nb_seqs)
    fbs = [initialize_forward_backward(hmm, obs_seqs[k]) for k in eachindex(obs_seqs)]
    forward_backward!(fbs, hmm, obs_seqs, nb_seqs)
    return [(fb.γ, fb.logL[]) for fb in fbs]
end

function forward_backward(hmm::AbstractHMM, obs_seq::Vector)
    return only(forward_backward(hmm, [obs_seq], 1))
end
