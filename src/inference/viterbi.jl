function viterbi!(q, δₜ, δₜ₋₁, ψ, b, p, A, op, obs_seq)
    N, T = length(p), length(obs_seq)
    likelihoods_vec!(b, op, obs_seq[1])
    δₜ .= p .* b
    δₜ₋₁ .= δₜ
    @views ψ[:, 1] .= zero(eltype(ψ))
    for t in 2:T
        likelihoods_vec!(b, op, obs_seq[t])
        for j in 1:N
            i_max = argmax(δₜ₋₁[i] * A[i, j] for i in 1:N)
            ψ[j, t] = i_max
            δₜ[j] = δₜ₋₁[i_max] * A[i_max, j] * b[j]
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

Apply the Viterbi algorithm to compute the most likely sequence of states of an HMM, and return it as a vector of integers.
"""
function viterbi(hmm::HMM, obs_seq)
    T, N = length(obs_seq), length(hmm)
    p = initial_distribution(hmm.state_process)
    A = transition_matrix(hmm.state_process)
    b = likelihoods_vec(hmm.obs_process, obs_seq[1])

    R = promote_type(eltype(p), eltype(A), eltype(b))
    δₜ = Vector{R}(undef, N)
    δₜ₋₁ = Vector{R}(undef, N)
    ψ = Matrix{Int}(undef, N, T)
    q = Vector{Int}(undef, T)

    viterbi!(q, δₜ, δₜ₋₁, ψ, b, p, A, hmm.obs_process, obs_seq)
    return q
end
