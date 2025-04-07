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
    logα::Matrix{R}
    logB::Matrix{R}
    tmp::Vector{R}
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
    logB::Matrix{R}
    α::Matrix{R}
    logα::Matrix{R}
    logβ::Matrix{R}
    logγ::Matrix{R}
    logξ::Vector{M}
    tmp::Vector{R}
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
    logα = Matrix{R}(undef, N, T)
    logB = Matrix{R}(undef, N, T)
    tmp = Vector{R}(undef, N)
    return ForwardStorage(α, logL, logα, logB, tmp)
end

function _forward!(
    storage::ForwardOrForwardBackwardStorage,
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer;
    error_if_not_finite::Bool,
)
    (; α, logα, logB, logL, tmp) = storage
    t1, t2 = seq_limits(seq_ends, k)
    for t in t1:t2
        logαₜ = view(logα, :, t)
        logBₜ = view(logB, :, t)
        obs_logdensities!(logBₜ, hmm, obs_seq[t], control_seq[t]; error_if_not_finite)
        if t == t1
            logαₜ .= log_initialization(hmm) .+ logBₜ
        else
            logαₜ₋₁ = view(logα, :, t - 1)
            logtrans = log_transition_matrix(hmm, control_seq[t - 1])
            for j in eachindex(logαₜ)
                tmp .= logαₜ₋₁ .+ view(logtrans, :, j)
                logαₜ[j] = logsumexp(tmp) + logBₜ[j]
            end
        end
    end
    logL[k] = logsumexp(view(logα, :, t2))
    for t in t1:t2
        α[:, t1:t2] .= exp.(view(logα, :, t) .- logsumexp(view(logα, :, t)))
    end

    error_if_not_finite && @argcheck isfinite(logL[k])
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
    error_if_not_finite::Bool=true,
)
    if seq_ends isa NTuple{1}
        for k in eachindex(seq_ends)
            _forward!(storage, hmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite)
        end
    else
        @threads for k in eachindex(seq_ends)
            _forward!(storage, hmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite)
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
    error_if_not_finite::Bool=true,
)
    storage = initialize_forward(hmm, obs_seq, control_seq; seq_ends)
    forward!(storage, hmm, obs_seq, control_seq; seq_ends, error_if_not_finite)
    return storage.α, storage.logL
end
