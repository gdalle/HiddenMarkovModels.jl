function forward!(αₜ, αₜ₊₁, logb, init, trans, dists, hmm::AbstractHMM, obs_seq)
    T = length(obs_seq)
    obs_distributions!(dists, hmm, 1)
    logb .= logdensityof.(dists, Ref(obs_seq[1]))
    logm = maximum(logb)
    αₜ .= init .* exp.(logb .- logm)
    c = inv(sum(αₜ))
    αₜ .*= c
    logL = -log(c) + logm
    for t in 1:(T - 1)
        transition_matrix!(trans, hmm, t)
        obs_distributions!(dists, hmm, t + 1)
        logb .= logdensityof.(dists, Ref(obs_seq[t + 1]))
        logm = maximum(logb)
        mul!(αₜ₊₁, trans', αₜ)
        αₜ₊₁ .*= exp.(logb .- logm)
        c = inv(sum(αₜ₊₁))
        αₜ₊₁ .*= c
        αₜ .= αₜ₊₁
        logL += -log(c) + logm
    end
    return logL
end

"""
    forward(hmm, obs_seq)

Apply the forward algorithm to an HMM.
    
Return a tuple `(α, logL)` where

- `logL` is the loglikelihood of the sequence
- `α[i]` is the posterior probability of state `i` at the end of the sequence.
"""
function forward(hmm::AbstractHMM, obs_seq)
    N = length(hmm)
    init = initialization(hmm)
    trans = transition_matrix(hmm)
    dists = obs_distributions(hmm)
    logb = loglikelihoods_vec(hmm, obs_seq[1])

    R = promote_type(eltype(init), eltype(trans), eltype(logb))
    αₜ = Vector{R}(undef, N)
    αₜ₊₁ = Vector{R}(undef, N)
    logL = forward!(αₜ, αₜ₊₁, logb, init, trans, dists, hmm, obs_seq)
    return αₜ, logL
end

"""
    forward(hmm, obs_seqs, nb_seqs)

Apply the forward algorithm to an HMM, based on multiple observation sequences.

Return a vector of tuples `(αₖ, logLₖ)`, where

- `logLₖ` is the loglikelihood of sequence `k`
- `αₖ[i]` is the posterior probability of state `i` at the end of sequence `k`

!!! warning "Multithreading"
    This function is parallelized across sequences.
"""
function forward(hmm::AbstractHMM, obs_seqs, nb_seqs::Integer)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    f1 = forward(hmm, first(obs_seqs))
    fs = Vector{typeof(f1)}(undef, nb_seqs)
    fs[1] = f1
    @threads for k in 2:nb_seqs
        fs[k] = forward(hmm, obs_seqs[k])
    end
    return fs
end

"""
    logdensityof(hmm, obs_seq)

Apply the forward algorithm to compute the loglikelihood of a single observation sequence for an HMM.

Return a number.
"""
function DensityInterface.logdensityof(hmm::AbstractHMM, obs_seq)
    return last(forward(hmm, obs_seq))
end

"""
    logdensityof(hmm, obs_seqs, nb_seqs)

Apply the forward algorithm to compute the total loglikelihood of multiple observation sequences for an HMM.

Return a number.

!!! warning "Multithreading"
    This function is parallelized across sequences.
"""
function DensityInterface.logdensityof(hmm::AbstractHMM, obs_seqs, nb_seqs::Integer)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    logL1 = logdensityof(hmm, first(obs_seqs))
    logLs = Vector{typeof(logL1)}(undef, nb_seqs)
    logLs[1] = logL1
    @threads for k in 2:nb_seqs
        logLs[k] = logdensityof(hmm, obs_seqs[k])
    end
    return sum(logLs)
end
