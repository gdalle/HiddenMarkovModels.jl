"""
$(TYPEDEF)

Store Viterbi quantities with element type `R`.

This storage is relative to a single sequence.

# Fields

The only field useful outside of the algorithm is `q`, the rest does not belong to the public API.

$(TYPEDFIELDS)
"""
struct ViterbiStorage{R}
    logL::RefValue{R}
    "observation loglikelihoods at a given time step"
    logb::Vector{R}
    "highest path score when accounting for the first `t` observations and ending at a given state"
    ϕ::Vector{R}
    "same as `ϕ` but for the previous time step"
    ϕ_prev::Vector{R}
    "penultimate state maximizing the path score"
    ψ::Matrix{Int}
    "most likely state at each time `q[t] = argmaxᵢ ℙ(X[t]=i | Y[1:T])`"
    q::Vector{Int}
    "scratch storage space"
    scratch::Vector{R}
end

"""
    initialize_viterbi(hmm, obs_seq)
    initialize_viterbi(hmm, MultiSeq(obs_seqs))
"""
function initialize_viterbi(
    hmm::AbstractHMM, obs_seq::Vector, control_seq::AbstractVector=no_controls(obs_seq)
)
    N, T = length(hmm), length(eachindex(obs_seq, control_seq))
    R = eltype(hmm, obs_seq[1], control_seq[1])
    logL = RefValue{R}()
    logb = Vector{R}(undef, N)
    ϕ = Vector{R}(undef, N)
    ϕ_prev = Vector{R}(undef, N)
    ψ = Matrix{Int}(undef, N, T)
    q = Vector{Int}(undef, T)
    scratch = Vector{R}(undef, N)
    return ViterbiStorage(logL, logb, ϕ, ϕ_prev, ψ, q, scratch)
end

function initialize_viterbi(
    hmm::AbstractHMM, obs_seqs::MultiSeq, control_seqs::MultiSeq == no_controls(obs_seqs)
)
    R = eltype(hmm, obs_seqs[1][1], control_seqs[1][1])
    storages = Vector{ViterbiStorage{R}}(undef, length(obs_seqs))
    for k in eachindex(storages, sequences(obs_seqs), sequences(control_seqs))
        storages[k] = initialize_viterbi(hmm, obs_seqs[k], control_seqs[k])
    end
    return storages
end

"""
    viterbi!(storage, hmm, obs_seq)
    viterbi!(storage, hmm, MultiSeq(obs_seqs))
"""
function viterbi!(
    storage::ViterbiStorage,
    hmm::AbstractHMM,
    obs_seq::Vector,
    control_seq::AbstractVector=no_controls(obs_seq),
)
    N, T = length(hmm), length(eachindex(obs_seq, control_seq))
    @unpack logL, logb, ϕ, ϕ_prev, ψ, q, scratch = storage
    init = initialization(hmm)
    obs_logdensities!(logb, hmm, obs_seq[1], control_seq[1])
    ϕ .= log.(init) .+ logb
    ϕ_prev .= ϕ
    for t in 2:T
        trans = transition_matrix(hmm, control_seq[t - 1])
        obs_logdensities!(logb, hmm, obs_seq[t], control_seq[t])
        for j in 1:N
            @views scratch .= ϕ_prev .+ log.(trans[:, j])
            i_max = argmax(scratch)
            ψ[j, t] = i_max
            ϕ[j] = scratch[i_max] + logb[j]
        end
        ϕ_prev .= ϕ
    end
    check_finite(ϕ)
    q[T] = argmax(ϕ)
    for t in (T - 1):-1:1
        q[t] = ψ[q[t + 1], t + 1]
    end
    logL[] = ϕ[q[T]]
    return nothing
end

function viterbi!(
    storages::Vector{<:ViterbiStorage},
    hmm::AbstractHMM,
    obs_seqs::MultiSeq,
    control_seqs::MultiSeq=no_controls(obs_seqs),
)
    for k in eachindex(storages, sequences(obs_seqs), sequences(control_seq))
        viterbi!(storages[k], hmm, obs_seqs[k], control_seqs[k])
    end
end

"""
    viterbi(hmm, obs_seq)
    viterbi(hmm, MultiSeq(obs_seq))

Apply the Viterbi algorithm to infer the most likely state sequence corresponding to `obs_seq` for `hmm`.

This function returns a tuple `(q, logL)` where `q` is a vector of integers giving the most likely state sequence, while `logL` is the loglikelihood of that sequence.
"""
function viterbi(
    hmm::AbstractHMM, obs_seqs::MultiSeq, control_seqs::MultiSeq=no_controls(obs_seqs)
)
    storages = initialize_viterbi(hmm, obs_seqs, control_seqs)
    viterbi!(storages, hmm, obs_seqs, control_seqs)
    return storages
end

function viterbi(
    hmm::AbstractHMM, obs_seq::Vector, control_seq::AbstractVector=no_controls(obs_seq)
)
    return only(viterbi(hmm, MultiSeq([obs_seq]), MultiSeq([control_seq])))
end
