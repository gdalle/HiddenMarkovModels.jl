function viterbi!(best_state_seq::Vector, δ::Matrix, δA_tmp::Vector, ψ::Matrix, p, A, B)
    N, T = size(δ)
    @views δ[:, 1] .= p .* B[:, 1]
    @views ψ[:, 1] .= zero(eltype(ψ))
    for t in 2:T
        for j in 1:N
            @views δA_tmp .= δ[:, t - 1] .* A[:, j]
            i_max = argmax(δA_tmp)
            δ[j, t] = δA_tmp[i_max] * B[j, t]
            ψ[j, t] = i_max
        end
    end
    @views best_state_seq[T] = argmax(δ[:, T])
    for t in (T - 1):-1:1
        best_state_seq[t] = ψ[best_state_seq[t + 1], t + 1]
    end
    return nothing
end

function viterbi(hmm::HMM, obs_seq::Vector)
    T = length(obs_seq)
    N = nb_states(hmm)
    p = initial_distribution(hmm)
    A = transition_matrix(hmm)
    B = likelihoods(hmm, obs_seq)

    R = promote_type(eltype(p), eltype(A), eltype(B))
    δ = Matrix{R}(undef, N, T)
    δA_tmp = Vector{R}(undef, N)
    ψ = Matrix{Int}(undef, N, T)
    best_state_seq = Vector{Int}(undef, T)

    viterbi!(best_state_seq, δ, δA_tmp, ψ, p, A, B)
    return best_state_seq
end
