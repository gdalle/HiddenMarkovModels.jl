"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct ForwardStorage{R}
    "posterior last state marginals `α[i] = ℙ(X[T]=i | Y[1:T])`"
    α::Matrix{R}
    "one loglikelihood per observation sequence"
    logL::Vector{R}
    B::Matrix{R}
    c::Vector{R}
end

"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct ForwardBackwardStorage{R,M<:AbstractMatrix{R}}
    "posterior state marginals `γ[i,t] = ℙ(X[t]=i | Y[1:T])`"
    γ::Matrix{R}
    "posterior transition marginals `ξ[t][i,j] = ℙ(X[t]=i, X[t+1]=j | Y[1:T])`"
    ξ::Vector{M}
    "one loglikelihood per observation sequence"
    logL::Vector{R}
    B::Matrix{R}
    α::Matrix{R}
    c::Vector{R}
    β::Matrix{R}
    Bβ::Matrix{R}
end

Base.eltype(::ForwardBackwardStorage{R}) where {R} = R

const ForwardOrForwardBackwardStorage{R} = Union{
    ForwardStorage{R},ForwardBackwardStorage{R}
}

"""
$(SIGNATURES)
"""
function initialize_forward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    N, T, K = size(hmm, control_seq[1]), length(obs_seq), length(seq_ends)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    α = Matrix{R}(undef, N, T)
    logL = Vector{R}(undef, K)
    B = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    return ForwardStorage(α, logL, B, c)
end

function _forward_digest_observation!(
    current_state_marginals::AbstractVector{<:Real},
    current_obs_likelihoods::AbstractVector{<:Real},
    hmm::AbstractHMM,
    obs,
    control,
    prev_obs,
)
    a, b = current_state_marginals, current_obs_likelihoods

    obs_logdensities!(b, hmm, obs, control, prev_obs)
    logm = maximum(b)
    b .= exp.(b .- logm)

    a .*= b
    c = inv(sum(a))
    lmul!(c, a)

    logL = -log(c) + logm
    return c, logL
end

function _forward!(
    storage::ForwardOrForwardBackwardStorage,
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer,
)
    (; α, B, c, logL) = storage
    t1, t2 = seq_limits(seq_ends, k)
    logL[k] = zero(eltype(logL))
    for t in t1:t2
        αₜ = view(α, :, t)
        Bₜ = view(B, :, t)
        if t == t1
            copyto!(αₜ, initialization(hmm, control_seq[t]))
        else
            αₜ₋₁ = view(α, :, t - 1)
            predict_next_state!(αₜ, hmm, αₜ₋₁, control_seq[t]) # If `t` influences the transition from time `t` to `t+1` (and not from time `t-1` to `t`), then the associated control must be at `t+1`, right? If `control_seq[t-1]`, then we're using the control associated with the previous state and not the correct control, aren't we? The transition matrix would be P(X_{t}|X_{t-1},U_{t-1}) and not P(X_{t}|X_{t-1},U_{t}) as it should be. E.g., if `t == t1 + 1`, then `αₜ₋₁ = view(α, :, t1)` and the function would use the transition matrix P(X_{t1+1}|X_{t1},U_{t1}) instead of P(X_{t1+1}|X_{t1},U_{t1+1}). Same at logdensity.jl, line 37; forward_backward.jl, line 53.
        end
        prev_obs = t == t1 ? missing : previous_obs(hmm, obs_seq, t)
        cₜ, logLₜ = _forward_digest_observation!(
            αₜ, Bₜ, hmm, obs_seq[t], control_seq[t], prev_obs
        )
        c[t] = cₜ
        logL[k] += logLₜ
    end

    @argcheck isfinite(logL[k])
    return nothing
end

"""
$(SIGNATURES)
"""
function forward!(
    storage::ForwardOrForwardBackwardStorage,
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    if seq_ends isa NTuple{1}
        for k in eachindex(seq_ends)
            _forward!(storage, hmm, obs_seq, control_seq, seq_ends, k)
        end
    else
        @threads for k in eachindex(seq_ends)
            _forward!(storage, hmm, obs_seq, control_seq, seq_ends, k)
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the forward algorithm to infer the current state after sequence `obs_seq` for `hmm`.
    
Return a tuple `(storage.α, storage.logL)` where `storage` is of type [`ForwardStorage`](@ref).
"""
function forward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    storage = initialize_forward(hmm, obs_seq, control_seq; seq_ends)
    forward!(storage, hmm, obs_seq, control_seq; seq_ends)
    return storage.α, storage.logL
end
