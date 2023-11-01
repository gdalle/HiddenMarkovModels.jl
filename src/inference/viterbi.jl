function viterbi!(
    q, δₜ, δₜ₋₁, δtrans_tmp, ψ, logb, init, trans, dists, hmm::AbstractHMM, obs_seq
)
    N, T = length(hmm), length(obs_seq)
    obs_distributions!(dists, hmm, 1)
    logb .= logdensityof.(dists, Ref(obs_seq[1]))
    logm = maximum(logb)
    δₜ .= init .* exp.(logb .- logm)
    δₜ₋₁ .= δₜ
    @views ψ[:, 1] .= zero(eltype(ψ))
    for t in 2:T
        transition_matrix!(trans, hmm, t - 1)
        obs_distributions!(dists, hmm, t)
        logb .= logdensityof.(dists, Ref(obs_seq[t]))
        logm = maximum(logb)
        for j in 1:N  # TODO: vectorize this loop?
            @views δtrans_tmp .= δₜ₋₁ .* trans[:, j]
            i_max = argmax(δtrans_tmp)
            ψ[j, t] = i_max
            δₜ[j] = δtrans_tmp[i_max] * exp(logb[j] - logm)
        end
        δₜ₋₁ .= δₜ
    end
    q[T] = argmax(δₜ)
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
    init = initialization(hmm)
    trans = transition_matrix(hmm)
    dists = obs_distributions(hmm)
    logb = loglikelihoods_vec(hmm, obs_seq[1])

    R = promote_type(eltype(init), eltype(trans), eltype(logb))
    δₜ = Vector{R}(undef, N)
    δₜ₋₁ = Vector{R}(undef, N)
    δtrans_tmp = Vector{R}(undef, N)
    ψ = Matrix{Int}(undef, N, T)
    q = Vector{Int}(undef, T)

    viterbi!(q, δₜ, δₜ₋₁, δtrans_tmp, ψ, logb, init, trans, dists, hmm, obs_seq)
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
