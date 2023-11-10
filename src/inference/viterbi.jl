"""
$(TYPEDEF)

Store Viterbi quantities with element type `R`.

This storage is relative to a single sequence.

# Fields

The only field useful outside of the algorithm is `q`.

$(TYPEDFIELDS)
"""
struct ViterbiStorage{R}
    "observation loglikelihoods at a given time step"
    logb::Vector{R}
    "highest path scores when accounting for the first `t` observations and ending at a given state"
    δ::Vector{R}
    "same as `δ` but for the previous time step"
    δ_prev::Vector{R}
    "penultimate state maximizing the path score"
    ψ::Matrix{Int}
    "most likely state at each time `q[t] = argmaxᵢ ℙ(X[t]=i | Y[1:T])`"
    q::Vector{Int}
    "scratch storage space"
    scratch::Vector{R}
end

function initialize_viterbi(hmm::AbstractHMM, obs_seq::Vector)
    T, N = length(obs_seq), length(hmm)
    R = eltype(hmm, obs_seq[1])

    logb = Vector{R}(undef, N)
    δ = Vector{R}(undef, N)
    δ_prev = Vector{R}(undef, N)
    ψ = Matrix{Int}(undef, N, T)
    q = Vector{Int}(undef, T)
    scratch = Vector{R}(undef, N)
    return ViterbiStorage(logb, δ, δ_prev, ψ, q, scratch)
end

function viterbi!(v::ViterbiStorage, hmm::AbstractHMM, obs_seq::Vector)
    N, T = length(hmm), length(obs_seq)
    p = initialization(hmm)
    A = transition_matrix(hmm)
    @unpack logb, δ, δ_prev, ψ, q, scratch = v

    obs_logdensities!(logb, hmm, obs_seq[1])
    logm = maximum(logb)
    δ .= p .* exp.(logb .- logm)
    δ_prev .= δ
    @views ψ[:, 1] .= zero(eltype(ψ))
    for t in 2:T
        obs_logdensities!(logb, hmm, obs_seq[t])
        logm = maximum(logb)
        for j in 1:N
            @views scratch .= δ_prev .* A[:, j]
            i_max = argmax(scratch)
            ψ[j, t] = i_max
            δ[j] = scratch[i_max] * exp(logb[j] - logm)
        end
        δ_prev .= δ
    end
    q[T] = argmax(δ)
    for t in (T - 1):-1:1
        q[t] = ψ[q[t + 1], t + 1]
    end
    return nothing
end

function viterbi!(
    vs::Vector{<:ViterbiStorage},
    hmm::AbstractHMM,
    obs_seqs::Vector{<:Vector},
    nb_seqs::Integer,
)
    check_lengths(obs_seqs, nb_seqs)
    @threads for k in eachindex(vs, obs_seqs)
        viterbi!(vs[k], hmm, obs_seqs[k])
    end
    return nothing
end

"""
    viterbi(hmm, obs_seq)
    viterbi(hmm, obs_seqs, nb_seqs)

Apply the Viterbi algorithm to infer the most likely state sequence of an HMM.

When applied on a single sequence, this function returns a vector of integers.
When applied on multiple sequences, it returns a vector of vectors of integers.
"""
function viterbi(hmm::AbstractHMM, obs_seqs::Vector{<:Vector}, nb_seqs::Integer)
    check_lengths(obs_seqs, nb_seqs)
    vs = [initialize_viterbi(hmm, obs_seqs[k]) for k in eachindex(obs_seqs)]
    viterbi!(vs, hmm, obs_seqs, nb_seqs)
    return [v.q for v in vs]
end

function viterbi(hmm::AbstractHMM, obs_seq::Vector)
    return only(viterbi(hmm, [obs_seq], 1))
end
