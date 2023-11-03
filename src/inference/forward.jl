"""
$(TYPEDEF)

Store forward quantities with element type `R`.

# Fields

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
    @unpack logb, αₜ, αₜ₊₁ = f

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

"""
    forward(hmm, obs_seq)

Apply the forward algorithm to an HMM.
    
Return a tuple `(α, logL)` where

- `logL` is the loglikelihood of the sequence
- `α[i]` is the posterior probability of state `i` at the end of the sequence.
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

- `logLₖ` is the loglikelihood of sequence `k`
- `αₖ[i]` is the posterior probability of state `i` at the end of sequence `k`

!!! warning "Multithreading"
    This function is parallelized across sequences.
"""
function forward(hmm::AbstractHMM, obs_seqs::Vector{<:Vector}, nb_seqs::Integer)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    R = eltype(hmm, obs_seqs[1][1])
    fs = Vector{ForwardStorage{R}}(undef, nb_seqs)
    @threads for k in eachindex(fs, obs_seqs)
        fs[k] = forward(hmm, obs_seqs[k])
    end
    return fs
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
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    R = eltype(hmm, obs_seqs[1][1])
    logLs = Vector{R}(undef, nb_seqs)
    @threads for k in eachindex(logLs, obs_seqs)
        logLs[k] = logdensityof(hmm, obs_seqs[k])
    end
    return sum(logLs)
end
