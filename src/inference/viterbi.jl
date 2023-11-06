"""
$(TYPEDEF)

Store Viterbi quantities with element type `R`.

This storage is relative to a single sequence.

# Fields

$(TYPEDFIELDS)
"""
struct ViterbiStorage{R}
    "vector of observation loglikelihoods `logb[i]`"
    logb::Vector{R}
    δₜ::Vector{R}
    δₜ₋₁::Vector{R}
    δₜ₋₁Aⱼ::Vector{R}
    ψ::Matrix{Int}
    "vector of most likely state at each time"
    q::Vector{Int}
end

function initialize_viterbi(hmm::AbstractHMM, obs_seq::Vector)
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

function viterbi!(v::ViterbiStorage, hmm::AbstractHMM, obs_seq::Vector)
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

function viterbi!(
    vs::Vector{<:ViterbiStorage}, hmm::AbstractHMM, obs_seqs::Vector{<:Vector}
)
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
function viterbi(hmm::AbstractHMM, obs_seq::Vector)
    v = initialize_viterbi(hmm, obs_seq)
    viterbi!(v, hmm, obs_seq)
    return v.q
end

function viterbi(hmm::AbstractHMM, obs_seqs::Vector{<:Vector}, nb_seqs::Integer)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    vs = [initialize_viterbi(hmm, obs_seqs[k]) for k in eachindex(obs_seqs)]
    viterbi!(vs, hmm, obs_seqs)
    return [v.q for v in vs]
end
