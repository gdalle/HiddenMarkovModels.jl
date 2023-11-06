"""
$(TYPEDEF)

Store forward quantities with element type `R`.

This storage is relative to a single sequence.

# Fields

These fields are not part of the public API.

$(TYPEDFIELDS)
"""
struct ForwardStorage{R}
    "total loglikelihood"
    logL::RefValue{R}
    "vector of observation loglikelihoods `logb[i]`"
    logb::Vector{R}
    "scaled forward variables `α[t]`"
    αₜ::Vector{R}
    "scaled forward variables `α[t+1]`"
    αₜ₊₁::Vector{R}
end

function initialize_forward(hmm::AbstractHMM, obs_seq::Vector)
    N = length(hmm)
    R = eltype(hmm, obs_seq[1])

    logL = RefValue{R}(zero(R))
    logb = Vector{R}(undef, N)
    αₜ = Vector{R}(undef, N)
    αₜ₊₁ = Vector{R}(undef, N)
    f = ForwardStorage(logL, logb, αₜ, αₜ₊₁)
    return f
end

function forward!(f::ForwardStorage, hmm::AbstractHMM, obs_seq::Vector)
    T = length(obs_seq)
    p = initialization(hmm)
    A = transition_matrix(hmm)
    d = obs_distributions(hmm)
    @unpack logL, logb, αₜ, αₜ₊₁ = f

    logb .= logdensityof.(d, (obs_seq[1],))
    logm = maximum(logb)
    αₜ .= p .* exp.(logb .- logm)
    c = inv(sum(αₜ))
    αₜ .*= c
    logL[] = -log(c) + logm
    for t in 1:(T - 1)
        logb .= logdensityof.(d, (obs_seq[t + 1],))
        logm = maximum(logb)
        mul!(αₜ₊₁, A', αₜ)
        αₜ₊₁ .*= exp.(logb .- logm)
        c = inv(sum(αₜ₊₁))
        αₜ₊₁ .*= c
        αₜ .= αₜ₊₁
        logL[] += -log(c) + logm
    end
    return nothing
end

function forward!(
    fs::Vector{<:ForwardStorage}, hmm::AbstractHMM, obs_seqs::Vector{<:Vector}
)
    @threads for k in eachindex(fs, obs_seqs)
        forward!(fs[k], hmm, obs_seqs[k])
    end
    return nothing
end

"""
    forward(hmm, obs_seq)

Apply the forward algorithm to an HMM.
    
Return a tuple `(α, logL)` where

- `α[i]` is the posterior probability of state `i` at the end of the sequence
- `logL` is the loglikelihood of the sequence
"""
function forward(hmm::AbstractHMM, obs_seq::Vector)
    f = initialize_forward(hmm, obs_seq)
    forward!(f, hmm, obs_seq)
    return f.αₜ, f.logL[]
end

"""
    forward(hmm, obs_seqs, nb_seqs)

Apply the forward algorithm to an HMM, based on multiple observation sequences.

Return a vector of tuples `(αₖ, logLₖ)`, where

- `αₖ[i]` is the posterior probability of state `i` at the end of sequence `k`
- `logLₖ` is the loglikelihood of sequence `k`

!!! warning "Multithreading"
    This function is parallelized across sequences.
"""
function forward(hmm::AbstractHMM, obs_seqs::Vector{<:Vector}, nb_seqs::Integer)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    fs = [initialize_forward(hmm, obs_seqs[k]) for k in eachindex(obs_seqs)]
    forward!(fs, hmm, obs_seqs)
    return [(f.αₜ, f.logL[]) for f in fs]
end

"""
    logdensityof(hmm, obs_seq)

Apply the forward algorithm to compute the loglikelihood of a single observation sequence for an HMM.

Return a number.
"""
function DensityInterface.logdensityof(hmm::AbstractHMM, obs_seq::Vector)
    _, logL = forward(hmm, obs_seq)
    return logL
end

"""
    logdensityof(hmm, obs_seqs, nb_seqs)

Apply the forward algorithm to compute the total loglikelihood of multiple observation sequences for an HMM.

Return a number.

!!! warning "Multithreading"
    This function is parallelized across sequences.
"""
function DensityInterface.logdensityof(
    hmm::AbstractHMM, obs_seqs::Vector{<:Vector}, nb_seqs::Integer
)
    logαs_and_logLs = forward(hmm, obs_seqs, nb_seqs)
    return sum(last, logαs_and_logLs)
end
