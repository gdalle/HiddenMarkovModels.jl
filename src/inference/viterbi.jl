"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct ViterbiStorage{R}
    "most likely state sequence `q[t] = argmaxᵢ ℙ(X[t]=i | Y[1:T])`"
    q::Vector{Int}
    "one joint loglikelihood per pair of observation sequence and most likely state sequence"
    logL::Vector{R}
    logB::Matrix{R}
    ϕ::Matrix{R}
    ψ::Matrix{Int}
end

Base.eltype(::ViterbiStorage{R}) where {R} = R

"""
$(SIGNATURES)
"""
function initialize_viterbi(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVector{Int},
)
    N, T, K = length(hmm), length(obs_seq), length(seq_ends)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    q = Vector{Int}(undef, T)
    logL = Vector{R}(undef, K)
    logB = Matrix{R}(undef, N, T)
    ϕ = Matrix{R}(undef, N, T)
    ψ = Matrix{Int}(undef, N, T)
    return ViterbiStorage(q, logL, logB, ϕ, ψ)
end

"""
$(SIGNATURES)
"""
function viterbi!(
    storage::ViterbiStorage{R},
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    t1::Integer,
    t2::Integer;
) where {R}
    (; q, logB, ϕ, ψ) = storage

    logBₜ₁ = view(logB, :, t1)
    obs_logdensities!(logBₜ₁, hmm, obs_seq[t1], control_seq[t1])
    loginit = log_initialization(hmm)
    ϕ[:, t1] .= loginit .+ logBₜ₁

    for t in (t1 + 1):t2
        logBₜ = view(logB, :, t)
        obs_logdensities!(logBₜ, hmm, obs_seq[t], control_seq[t])
        logtrans = log_transition_matrix(hmm, control_seq[t - 1])
        ϕₜ, ϕₜ₋₁ = view(ϕ, :, t), view(ϕ, :, t - 1)
        ψₜ = view(ψ, :, t)
        argmaxplus_mul!(ϕₜ, ψₜ, logtrans, ϕₜ₋₁)
        ϕₜ .+= logBₜ
    end

    ϕₜ₂ = view(ϕ, :, t2)
    q[t2] = argmax(ϕₜ₂)
    logL = ϕ[q[t2], t2]
    for t in (t2 - 1):-1:t1
        q[t] = ψ[q[t + 1], t + 1]
    end

    @argcheck isfinite(logL)
    return logL
end

"""
$(SIGNATURES)
"""
function viterbi!(
    storage::ViterbiStorage{R},
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVector{Int},
) where {R}
    (; logL) = storage
    @threads for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        logL[k] = viterbi!(storage, hmm, obs_seq, control_seq, t1, t2;)
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the Viterbi algorithm to infer the most likely state sequence corresponding to `obs_seq` for `hmm`.

Return a tuple `(storage.q, storage.logL)` where `storage` is of type [`ViterbiStorage`](@ref).
"""
function viterbi(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVector{Int}=Fill(length(obs_seq), 1),
)
    storage = initialize_viterbi(hmm, obs_seq, control_seq; seq_ends)
    viterbi!(storage, hmm, obs_seq, control_seq; seq_ends)
    return storage.q, storage.logL
end
