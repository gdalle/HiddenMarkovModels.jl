function viterbi!(q, δₜ, δₜ₋₁, ψ, logb, p, A, op::ObservationProcess, obs_seq)
    N, T = length(op), length(obs_seq)
    loglikelihoods_vec!(logb, op, obs_seq[1])
    m = maximum(logb)
    δₜ .= p .* exp.(logb .- m)
    δₜ₋₁ .= δₜ
    @views ψ[:, 1] .= zero(eltype(ψ))
    for t in 2:T
        loglikelihoods_vec!(logb, op, obs_seq[t])
        m = maximum(logb)
        for j in 1:N
            i_max = argmax(δₜ₋₁[i] * A[i, j] for i in 1:N)
            ψ[j, t] = i_max
            δₜ[j] = δₜ₋₁[i_max] * A[i_max, j] * exp(logb[j] - m)
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

Apply the Viterbi algorithm to compute the most likely state sequence of an HMM for a single observation sequence.
"""
function viterbi(hmm::HMM, obs_seq)
    T, N = length(obs_seq), length(hmm)
    p = initial_distribution(hmm.state_process)
    A = transition_matrix(hmm.state_process)
    logb = loglikelihoods_vec(hmm.obs_process, obs_seq[1])

    R = promote_type(eltype(p), eltype(A), eltype(logb))
    δₜ = Vector{R}(undef, N)
    δₜ₋₁ = Vector{R}(undef, N)
    ψ = Matrix{Int}(undef, N, T)
    q = Vector{Int}(undef, T)

    viterbi!(q, δₜ, δₜ₋₁, ψ, logb, p, A, hmm.obs_process, obs_seq)
    return q
end

"""
    viterbi(hmm, obs_seqs, nb_seqs)

Apply the Viterbi algorithm to compute the most likely state sequences of an HMM for multiple observation sequences.
"""
function viterbi(hmm::HMM, obs_seqs, nb_seqs::Integer)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    qs = Vector{Vector{Int}}(undef, nb_seqs)
    @threads for k in 1:nb_seqs
        qs[k] = viterbi(hmm, obs_seqs[k])
    end
    return qs
end
