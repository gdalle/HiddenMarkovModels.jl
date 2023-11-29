"""
$(TYPEDEF)

Store forward quantities with element type `R`.

This storage is relative to a single sequence.

# Fields

The only fields useful outside of the algorithm are `α` and `logL`, the rest does not belong to the public API.

$(TYPEDFIELDS)
"""
struct ForwardStorage{R}
    "total loglikelihood of the observation sequence"
    logL::RefValue{R}
    "observation loglikelihoods `logbₜ[i] = ℙ(Y[t] | X[t]=i)`"
    logb::Vector{R}
    "scaled forward messsages for a given time step"
    α::Vector{R}
    "same as `α` but for the next time step"
    α_next::Vector{R}
end

"""
    initialize_forward(hmm, obs_seq)
"""
function initialize_forward(hmm::AbstractHMM, obs_seq::Vector)
    N = length(hmm)
    R = eltype(hmm, obs_seq[1])

    logL = RefValue{R}(zero(R))
    logb = Vector{R}(undef, N)
    α = Vector{R}(undef, N)
    α_next = Vector{R}(undef, N)
    storage = ForwardStorage(logL, logb, α, α_next)
    return storage
end

"""
    forward!(storage, hmm, obs_seq)
"""
function forward!(storage::ForwardStorage, hmm::AbstractHMM, obs_seq::Vector)
    T = length(obs_seq)
    p = initialization(hmm)
    @unpack logL, logb, α, α_next = storage

    obs_logdensities!(logb, hmm, 1, obs_seq[1])
    check_right_finite(logb)
    logm = maximum(logb)
    α .= p .* exp.(logb .- logm)
    c = inv(sum(α))
    α .*= c
    check_finite(α)
    logL[] = -log(c) + logm
    for t in 1:(T - 1)
        A = transition_matrix(hmm, t)
        obs_logdensities!(logb, hmm, t + 1, obs_seq[t + 1])
        check_right_finite(logb)
        logm = maximum(logb)
        mul!(α_next, A', α)
        α_next .*= exp.(logb .- logm)
        c = inv(sum(α_next))
        α_next .*= c
        α .= α_next
        check_finite(α)
        logL[] += -log(c) + logm
    end
    return nothing
end

"""
    forward(hmm, obs_seq)

Run the forward algorithm to infer the current state of `hmm` after sequence `obs_seq`.
    
This function returns a tuple `(α, logL)` where

- `α[i]` is the posterior probability of state `i` at the end of the sequence
- `logL` is the loglikelihood of the sequence
"""
function forward(hmm::AbstractHMM, obs_seq::Vector)
    storage = initialize_forward(hmm, obs_seq)
    forward!(storage, hmm, obs_seq)
    return (α=storage.α, logL=storage.logL[])
end

"""
    logdensityof(hmm, obs_seq)

Run the forward algorithm to compute the posterior loglikelihood of sequence `obs_seq` for `hmm`.

This function returns a number.
"""
function DensityInterface.logdensityof(hmm::AbstractHMM, obs_seq::Vector)
    _, logL = forward(hmm, obs_seq)
    return logL
end
