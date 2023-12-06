"""
$(TYPEDEF)

Store forward quantities with element type `R`.

This storage is relative to a single sequence.

# Fields

The only fields useful outside of the algorithm are `α` and `logL`, the rest does not belong to the public API.

$(TYPEDFIELDS)
"""
struct ForwardStorage{R}
    logL::RefValue{R}
    "observation loglikelihoods `logb[i] = ℙ(Y[t] | X[t]=i)`"
    logb::Vector{R}
    "scaled forward messsages for a given time step"
    α::Vector{R}
    "same as `α` but for the next time step"
    α_next::Vector{R}
end

"""
    initialize_forward(hmm, obs_seq)
    initialize_forward(hmm, MultiSeq(obs_seqs))
"""
function initialize_forward(
    hmm::AbstractHMM, obs_seq::AbstractVector, control_seq::AbstractVector
)
    N = length(hmm)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    logL = RefValue{R}()
    logb = Vector{R}(undef, N)
    α = Vector{R}(undef, N)
    α_next = Vector{R}(undef, N)
    storage = ForwardStorage(logL, logb, α, α_next)
    return storage
end

function initialize_forward(hmm::AbstractHMM, obs_seqs::MultiSeq, control_seqs::MultiSeq)
    R = eltype(hmm, obs_seqs[1][1], control_seqs[1][1])
    storages = Vector{ForwardStorage{R}}(undef, length(obs_seqs))
    for k in eachindex(storages, sequences(obs_seqs), sequences(control_seqs))
        storages[k] = initialize_forward(hmm, obs_seqs[k], control_seqs[k])
    end
    return storages
end

"""
    forward!(storage, hmm, obs_seq)
    forward!(storages, hmm, MultiSeq(obs_seqs))
"""
function forward!(
    storage::ForwardStorage,
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
)
    T = length(obs_seq)
    @unpack logL, logb, α, α_next = storage

    init = initialization(hmm)
    obs_logdensities!(logb, hmm, obs_seq[1], control_seq[1])
    logm = maximum(logb)
    α .= init .* exp.(logb .- logm)
    c = inv(sum(α))
    α .*= c
    logL[] = -log(c) + logm

    for t in 1:(T - 1)
        trans = transition_matrix(hmm, control_seq[t])
        obs_logdensities!(logb, hmm, obs_seq[t + 1], control_seq[t + 1])
        logm = maximum(logb)
        mul!(α_next, trans', α)
        α_next .*= exp.(logb .- logm)
        c = inv(sum(α_next))
        α_next .*= c
        α .= α_next
        logL[] += -log(c) + logm
    end

    check_finite(α)
    return nothing
end

function forward!(
    storages::Vector{<:ForwardStorage},
    hmm::AbstractHMM,
    obs_seqs::MultiSeq,
    control_seqs::MultiSeq,
)
    for k in eachindex(storages, sequences(obs_seqs), sequences(control_seqs))
        forward!(storages[k], hmm, obs_seqs[k], control_seqs[k])
    end
end

"""
    forward(hmm, obs_seq)
    forward(hmm, MultiSeq(obs_seqs))

Run the forward algorithm to infer the current state of `hmm` after sequence `obs_seq`.
    
This function returns a tuple `(α, logL)` where

- `α[i]` is the posterior probability of state `i` at the end of the sequence
- `logL` is the loglikelihood of the sequence
"""
function forward(
    hmm::AbstractHMM, obs_seqs::MultiSeq, control_seqs::MultiSeq=no_controls(obs_seqs)
)
    storages = initialize_forward(hmm, obs_seqs, control_seqs)
    forward!(storages, hmm, obs_seqs, control_seqs)
    return map(result, storages)
end

function forward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=no_controls(obs_seq),
)
    return only(forward(hmm, MultiSeq([obs_seq]), MultiSeq([control_seq])))
end

result(storage::ForwardStorage) = (storage.α, storage.logL[])
