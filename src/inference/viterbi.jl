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
    logL::RefValue{R}
    logb::Vector{R}
    ϕ::Vector{R}
    ϕ_prev::Vector{R}
    ψ::Matrix{Int}
    scratch::Vector{R}
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
    N, T = length(hmm), length(obs_seq)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    q = Vector{Int}(undef, T)
    logL = RefValue{R}()
    logb = Vector{R}(undef, N)
    ϕ = Vector{R}(undef, N)
    ϕ_prev = Vector{R}(undef, N)
    ψ = Matrix{Int}(undef, N, T)
    scratch = Vector{R}(undef, N)
    return ViterbiStorage(q, logL, logb, ϕ, ϕ_prev, ψ, scratch)
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
    @unpack logL, logb, ϕ, ϕ_prev, ψ, q, scratch = storage
    logL[] = zero(R)
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        init = initialization(hmm)
        obs_logdensities!(logb, hmm, obs_seq[t1], control_seq[t1])
        ϕ .= log.(init) .+ logb
        ϕ_prev .= ϕ
        for t in (t1 + 1):t2
            trans = transition_matrix(hmm, control_seq[t - 1])
            obs_logdensities!(logb, hmm, obs_seq[t], control_seq[t])
            for j in 1:length(hmm)
                @views scratch .= ϕ_prev .+ log.(trans[:, j])
                i_max = argmax(scratch)
                ψ[j, t] = i_max
                ϕ[j] = scratch[i_max] + logb[j]
            end
            ϕ_prev .= ϕ
        end
        check_finite(ϕ)
        q[t2] = argmax(ϕ)
        for t in (t2 - 1):-1:t1
            q[t] = ψ[q[t + 1], t + 1]
        end
        logL[] += ϕ[q[t2]]
    end
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
    return storage.q, storage.logL[]
end
