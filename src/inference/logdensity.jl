function forward_light!(αₜ, αₜ₊₁, logb, p, A, hmm::AbstractHMM, obs_seq)
    T = length(obs_seq)
    loglikelihoods_vec!(logb, hmm, obs_seq[1])
    m = maximum(logb)
    αₜ .= p .* exp.(logb .- m)
    c = inv(sum(αₜ))
    αₜ .*= c
    logL = -log(c) + m
    for t in 1:(T - 1)
        loglikelihoods_vec!(logb, hmm, obs_seq[t + 1])
        m = maximum(logb)
        mul!(αₜ₊₁, A', αₜ)
        αₜ₊₁ .*= exp.(logb .- m)
        c = inv(sum(αₜ₊₁))
        αₜ₊₁ .*= c
        αₜ .= αₜ₊₁
        logL += -log(c) + m
    end
    return logL
end

"""
    logdensityof(hmm, obs_seq)

Apply the forward algorithm to compute the loglikelihood of a single observation sequence for an HMM.
"""
function DensityInterface.logdensityof(hmm::AbstractHMM, obs_seq)
    N = length(hmm)
    p = initial_distribution(hmm)
    A = transition_matrix(hmm)
    logb = loglikelihoods_vec(hmm, obs_seq[1])

    R = promote_type(eltype(p), eltype(A), eltype(logb))
    αₜ = Vector{R}(undef, N)
    αₜ₊₁ = Vector{R}(undef, N)
    logL = forward_light!(αₜ, αₜ₊₁, logb, p, A, hmm, obs_seq)
    return logL
end

"""
    logdensityof(hmm, obs_seqs, nb_seqs)

Apply the forward algorithm to compute the total loglikelihood of multiple observation sequences for an HMM.

!!! warning "Multithreading"
    This function is parallelized across sequences.
"""
function DensityInterface.logdensityof(hmm::AbstractHMM, obs_seqs, nb_seqs::Integer)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    logL1 = logdensityof(hmm, first(obs_seqs))
    logLs = Vector{typeof(logL1)}(undef, nb_seqs)
    @threads for k in 2:nb_seqs
        logLs[k] = logdensityof(hmm, obs_seqs[k])
    end
    return sum(logLs)
end
