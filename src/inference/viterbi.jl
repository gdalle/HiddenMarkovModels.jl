function viterbi!(q, δₜ, δₜ₋₁, δA_tmp, ψ, logb, p, A, hmm::AbstractHMM, obs_seq)
    N, T = length(hmm), length(obs_seq)
    loglikelihoods_vec!(logb, hmm, obs_seq[1])
    m = maximum(logb)
    δₜ .= p .* exp.(logb .- m)
    δₜ₋₁ .= δₜ
    @views ψ[:, 1] .= zero(eltype(ψ))
    for t in 2:T
        loglikelihoods_vec!(logb, hmm, obs_seq[t])
        m = maximum(logb)
        for j in 1:N
            @views δA_tmp .= δₜ₋₁ .* A[:, j]
            i_max = argmax(δA_tmp)
            ψ[j, t] = i_max
            δₜ[j] = δA_tmp[i_max] * exp(logb[j] - m)
        end
        δₜ₋₁ .= δₜ
    end
    @views q[T] = argmax(δₜ)
    for t in (T - 1):-1:1
        q[t] = ψ[q[t + 1], t + 1]
    end
    return nothing
end

"""
    viterbi(hmm, obs_seq)

Apply the Viterbi algorithm to compute the most likely state sequence of an HMM.

Return a vector of integers.
"""
function viterbi(hmm::AbstractHMM, obs_seq)
    T, N = length(obs_seq), length(hmm)
    p = initial_distribution(hmm)
    A = transition_matrix(hmm)
    logb = loglikelihoods_vec(hmm, obs_seq[1])

    R = promote_type(eltype(p), eltype(A), eltype(logb))
    δₜ = Vector{R}(undef, N)
    δₜ₋₁ = Vector{R}(undef, N)
    δA_tmp = Vector{R}(undef, N)
    ψ = Matrix{Int}(undef, N, T)
    q = Vector{Int}(undef, T)

    viterbi!(q, δₜ, δₜ₋₁, δA_tmp, ψ, logb, p, A, hmm, obs_seq)
    return q
end

"""
    viterbi(hmm, obs_seqs, nb_seqs)

Apply the Viterbi algorithm to compute the most likely state sequences of an HMM, based on multiple observation sequences.

Return a vector of vectors of integers.

!!! warning "Multithreading"
    This function is parallelized across sequences.
"""
function viterbi(hmm::AbstractHMM, obs_seqs, nb_seqs::Integer)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    qs = Vector{Vector{Int}}(undef, nb_seqs)
    @threads for k in 1:nb_seqs
        qs[k] = viterbi(hmm, obs_seqs[k])
    end
    return qs
end
