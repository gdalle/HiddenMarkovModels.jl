"""
$(TYPEDEF)

Store forward quantities with element type `R`.

# Fields

Let `X` denote the vector of hidden states and `Y` denote the vector of observations.

$(TYPEDFIELDS)
"""
struct ForwardStorage{R}
    "vector of observation loglikelihoods `logb[i]`"
    logb::Vector{R}
    "scaled forward variables `α[t]` proportional to `ℙ(Y[1:t], X[t]=i)` (up to a function of `t`)"
    αₜ::Vector{R}
    "scaled forward variables `α[t+1]`"
    αₜ₊₁::Vector{R}
end

function initialize_forward(hmm::AbstractHMM, obs_seq)
    N = length(hmm)
    p = initialization(hmm)
    A = transition_matrix(hmm)
    d = obs_distributions(hmm)
    testval = logdensityof(d[1], obs_seq[1])
    R = promote_type(eltype(p), eltype(A), typeof(testval))

    logb = Vector{R}(undef, N)
    αₜ = Vector{R}(undef, N)
    αₜ₊₁ = Vector{R}(undef, N)
    f = ForwardStorage(logb, αₜ, αₜ₊₁)
    return f
end

function forward!(f::ForwardStorage, hmm::AbstractHMM, obs_seq)
    T = length(obs_seq)
    p = initialization(hmm)
    A = transition_matrix(hmm)
    d = obs_distributions(hmm)
    @unpack logb, αₜ, αₜ₊₁ = f

    logb .= logdensityof.(d, Ref(obs_seq[1]))
    logm = maximum(logb)
    αₜ .= p .* exp.(logb .- logm)
    c = inv(sum(αₜ))
    αₜ .*= c
    logL = -log(c) + logm
    for t in 1:(T - 1)
        logb .= logdensityof.(d, Ref(obs_seq[t + 1]))
        logm = maximum(logb)
        mul!(αₜ₊₁, A', αₜ)
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
    f = initialize_forward(hmm, obs_seq)
    logL = forward!(f, hmm, obs_seq)
    return f.αₜ, logL
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
    @threads for k in eachindex(fs, obs_seqs)
        if k > 1
            fs[k] = forward(hmm, obs_seqs[k])
        end
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
    @threads for k in eachindex(logLs, obs_seqs)
        if k > 1
            logLs[k] = logdensityof(hmm, obs_seqs[k])
        end
    end
    return sum(logLs)
end
