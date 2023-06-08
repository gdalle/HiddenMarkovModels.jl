function viterbi!(q, δₜ, δₜ₋₁, ψ, b, p, A, op::ObservationProcess, obs_seq)
    N, T = length(op), length(obs_seq)
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

function viterbi_semilog!(q, δₜ, δₜ₋₁, ψ, logb, p, A, op::ObservationProcess, obs_seq)
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

function viterbi_log!(
    q, logδₜ, logδₜ₋₁, ψ, logb, logp, logA, op::ObservationProcess, obs_seq
)
    N, T = length(op), length(obs_seq)
    loglikelihoods_vec!(logb, op, obs_seq[1])
    logδₜ .= logp .+ logb
    logδₜ₋₁ .= logδₜ
    @views ψ[:, 1] .= zero(eltype(ψ))
    for t in 2:T
        loglikelihoods_vec!(logb, op, obs_seq[t])
        for j in 1:N
            i_max = argmax(logδₜ₋₁[i] + logA[i, j] for i in 1:N)
            ψ[j, t] = i_max
            logδₜ[j] = logδₜ₋₁[i_max] + logA[i_max, j] + logb[j]
        end
        logδₜ₋₁ .= logδₜ
    end
    @views q[T] = argmax(logδₜ)
    for t in (T - 1):-1:1
        q[t] = ψ[q[t + 1], t + 1]
    end
    return nothing
end

"""
    viterbi(hmm, obs_seq, scale=LogScale())

Apply the Viterbi algorithm to compute the most likely sequence of states of an HMM.
"""
function viterbi(hmm::HMM, obs_seq)
    return viterbi(hmm, obs_seq, LogScale())
end

function viterbi(hmm::HMM, obs_seq, ::NormalScale)
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

function viterbi(hmm::HMM, obs_seq, ::SemiLogScale)
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

function viterbi(hmm::HMM, obs_seq, ::LogScale)
    T, N = length(obs_seq), length(hmm)
    logp = log_initial_distribution(hmm.state_process)
    logA = log_transition_matrix(hmm.state_process)
    logb = loglikelihoods_vec(hmm.obs_process, obs_seq[1])

    R = promote_type(eltype(logp), eltype(logA), eltype(logb))
    logδₜ = Vector{R}(undef, N)
    logδₜ₋₁ = Vector{R}(undef, N)
    ψ = Matrix{Int}(undef, N, T)
    q = Vector{Int}(undef, T)

    viterbi_log!(q, logδₜ, logδₜ₋₁, ψ, logb, logp, logA, hmm.obs_process, obs_seq)
    return q
end
