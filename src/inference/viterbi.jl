"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct ViterbiStorage{R}
    "most likely state sequence `q[t] = argmaxᵢ ℙ(X[t]=i | Y[1:T])`"
    q::Vector{Int}
    "joint loglikelihood of the observation sequence & the most likely state sequence"
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
    obs_seq::AbstractVector;
    control_seq::AbstractVector,
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
    obs_seq::AbstractVector;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
) where {R}
    @unpack q, logL, logB, ϕ, ψ = storage

    @views for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)

        obs_logdensities!(logB[:, t1], hmm, obs_seq[t1], control_seq[t1])
        init = initialization(hmm)
        ϕ[:, t1] .= log.(init) .+ logB[:, t1]

        for t in (t1 + 1):t2
            obs_logdensities!(logB[:, t], hmm, obs_seq[t], control_seq[t])
            trans = transition_matrix(hmm, control_seq[t - 1])
            for j in 1:length(hmm)
                i_max = argmax(ϕ[i, t - 1] + log(trans[i, j]) for i in 1:length(hmm))
                ψ[j, t] = i_max
                ϕ[j, t] = ϕ[i_max, t - 1] + log(trans[i_max, j]) + logB[j, t]
            end
        end

        q[t2] = argmax(ϕ[:, t2])
        for t in (t2 - 1):-1:t1
            q[t] = ψ[q[t + 1], t + 1]
        end
        logL[k] = ϕ[q[t2], t2]
    end

    check_finite(ϕ)
    return nothing
end

"""
$(SIGNATURES)

Apply the Viterbi algorithm to infer the most likely state sequence corresponding to `obs_seq` for `hmm`.

Return a tuple `(q, logL)` defined in [`ViterbiStorage`](@ref).

# Keyword arguments

$(DESCRIBE_CONTROL_STARTS)
"""
function viterbi(
    hmm::AbstractHMM,
    obs_seq::AbstractVector;
    control_seq::AbstractVector=Fill(nothing, length(obs_seq)),
    seq_ends::AbstractVector{Int}=[length(obs_seq)],
)
    storage = initialize_viterbi(hmm, obs_seq; control_seq, seq_ends)
    viterbi!(storage, hmm, obs_seq; control_seq, seq_ends)
    return storage.q, sum(storage.logL)
end
