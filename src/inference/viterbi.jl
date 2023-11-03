"""
$(TYPEDEF)

Store Viterbi quantities with element type `R`.

# Fields

Let `X` denote the vector of hidden states and `Y` denote the vector of observations.

$(TYPEDFIELDS)
"""
struct ViterbiStorage{R}
    logb::Vector{R}
    δₜ::Vector{R}
    δₜ₋₁::Vector{R}
    δₜ₋₁Aⱼ::Vector{R}
    ψ::Matrix{Int}
    q::Vector{Int}
end

function initialize_viterbi(hmm::AbstractHMM, obs_seq)
    T, N = length(obs_seq), length(hmm)
    R = eltype(hmm, obs_seq[1])

    logb = Vector{R}(undef, N)
    δₜ = Vector{R}(undef, N)
    δₜ₋₁ = Vector{R}(undef, N)
    δₜ₋₁Aⱼ = Vector{R}(undef, N)
    ψ = Matrix{Int}(undef, N, T)
    q = Vector{Int}(undef, T)
    return ViterbiStorage(logb, δₜ, δₜ₋₁, δₜ₋₁Aⱼ, ψ, q)
end

function viterbi!(v::ViterbiStorage, hmm::AbstractHMM, obs_seq)
    N, T = length(hmm), length(obs_seq)
    p = initialization(hmm)
    A = transition_matrix(hmm)
    d = obs_distributions(hmm)
    @unpack logb, δₜ, δₜ₋₁, δₜ₋₁Aⱼ, ψ, q = v

    logb .= logdensityof.(d, (obs_seq[1],))
    logm = maximum(logb)
    δₜ .= p .* exp.(logb .- logm)
    δₜ₋₁ .= δₜ
    @views ψ[:, 1] .= zero(eltype(ψ))
    for t in 2:T
        logb .= logdensityof.(d, (obs_seq[t],))
        logm = maximum(logb)
        for j in 1:N
            @views δₜ₋₁Aⱼ .= δₜ₋₁ .* A[:, j]
            i_max = argmax(δₜ₋₁Aⱼ)
            ψ[j, t] = i_max
            δₜ[j] = δₜ₋₁Aⱼ[i_max] * exp(logb[j] - logm)
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
    v = initialize_viterbi(hmm, obs_seq)
    viterbi!(v, hmm, obs_seq)
    return v.q
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
    @threads for k in eachindex(qs, obs_seqs)
        qs[k] = viterbi(hmm, obs_seqs[k])
    end
    return qs
end
