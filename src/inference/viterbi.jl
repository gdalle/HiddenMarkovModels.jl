"""
$(TYPEDEF)

Store Viterbi quantities with element type `R`.

This storage is relative to a single sequence.

# Fields

The only field useful outside of the algorithm is `q`, the rest does not belong to the public API.

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

"""
    initialize_viterbi(hmm, obs_seq)
"""
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

"""
    viterbi!(storage, hmm, obs_seq)
"""
function viterbi!(storage::ViterbiStorage, hmm::AbstractHMM, obs_seq::Vector)
    N, T = length(hmm), length(obs_seq)
    p = initialization(hmm)
    A = transition_matrix(hmm)
    @unpack logb, δ, δ_prev, ψ, q, scratch = storage

    obs_logdensities!(logb, hmm, obs_seq[1])
    check_right_finite(logb)
    logm = maximum(logb)
    δ .= p .* exp.(logb .- logm)
    check_finite(δ)
    δ_prev .= δ
    @views ψ[:, 1] .= zero(eltype(ψ))
    for t in 2:T
        obs_logdensities!(logb, hmm, obs_seq[t])
        check_right_finite(logb)
        logm = maximum(logb)
        for j in 1:N
            @views scratch .= δ_prev .* A[:, j]
            i_max = argmax(scratch)
            ψ[j, t] = i_max
            δ[j] = scratch[i_max] * exp(logb[j] - logm)
        end
        check_finite(δ)
        δ_prev .= δ
    end
    q[T] = argmax(δ)
    for t in (T - 1):-1:1
        q[t] = ψ[q[t + 1], t + 1]
    end
    return nothing
end

"""
    viterbi(hmm, obs_seq)

Apply the Viterbi algorithm to infer the most likely state sequence corresponding to `obs_seq` for `hmm`.

This function returns a vector of integers.
"""
function viterbi(hmm::AbstractHMM, obs_seq::Vector)
    storage = initialize_viterbi(hmm, obs_seq)
    viterbi!(storage, hmm, obs_seq)
    return storage.q
end
