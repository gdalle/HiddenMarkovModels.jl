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
    N, T, K = length(hmm), length(obs_seq), length(seq_ends)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    α = Matrix{R}(undef, N, T)
    logL = Vector{R}(undef, K)
    B = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    return ForwardStorage(α, logL, B, c)
end

"""
$(SIGNATURES)
"""
function forward!(
    storage::ForwardOrForwardBackwardStorage,
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    t1::Integer,
    t2::Integer;
)
    (; α, B, c) = storage

    # Initialization
    Bₜ₁ = view(B, :, t1)
    obs_logdensities!(Bₜ₁, hmm, obs_seq[t1], control_seq[t1])
    logm = maximum(Bₜ₁)
    Bₜ₁ .= exp.(Bₜ₁ .- logm)

    init = initialization(hmm)
    αₜ₁ = view(α, :, t1)
    αₜ₁ .= init .* Bₜ₁
    c[t1] = inv(sum(αₜ₁))
    lmul!(c[t1], αₜ₁)

    logL = -log(c[t1]) + logm

    # Loop
    for t in t1:(t2 - 1)
        Bₜ₊₁ = view(B, :, t + 1)
        obs_logdensities!(Bₜ₊₁, hmm, obs_seq[t + 1], control_seq[t + 1])
        logm = maximum(Bₜ₊₁)
        Bₜ₊₁ .= exp.(Bₜ₊₁ .- logm)

        trans = transition_matrix(hmm, control_seq[t])
        αₜ, αₜ₊₁ = view(α, :, t), view(α, :, t + 1)
        mul!(αₜ₊₁, transpose(trans), αₜ)
        αₜ₊₁ .*= Bₜ₊₁
        c[t + 1] = inv(sum(αₜ₊₁))
        lmul!(c[t + 1], αₜ₊₁)

        logL += -log(c[t + 1]) + logm
    end

    @argcheck isfinite(logL)
    return logL
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
    (; logL) = storage
    if seq_ends isa NTuple
        for k in eachindex(seq_ends)
            t1, t2 = seq_limits(seq_ends, k)
            logL[k] = forward!(storage, hmm, obs_seq, control_seq, t1, t2;)
        end
    else
        @threads for k in eachindex(seq_ends)
            t1, t2 = seq_limits(seq_ends, k)
            logL[k] = forward!(storage, hmm, obs_seq, control_seq, t1, t2;)
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
