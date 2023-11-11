"""
$(TYPEDEF)

Store forward quantities with element type `R`.

This storage is relative to a single sequence.

# Fields

The only fields useful outside of the algorithm are `αₜ` and `logL`.

$(TYPEDFIELDS)
"""
struct ForwardStorage{R}
    "total loglikelihood"
    logL::RefValue{R}
    "observation loglikelihoods `logbₜ[i] = ℙ(Y[t] | X[t]=i)`"
    logb::Vector{R}
    "scaled forward messsages for a given time step"
    α::Vector{R}
    "same as `α` but for the next time step"
    α_next::Vector{R}
end

function initialize_forward(hmm::AbstractHMM, obs_seq::Vector)
    N = length(hmm)
    R = eltype(hmm, obs_seq[1])

    logL = RefValue{R}(zero(R))
    logb = Vector{R}(undef, N)
    α = Vector{R}(undef, N)
    α_next = Vector{R}(undef, N)
    f = ForwardStorage(logL, logb, α, α_next)
    return f
end

function forward!(f::ForwardStorage, hmm::AbstractHMM, obs_seq::Vector)
    T = length(obs_seq)
    p = initialization(hmm)
    A = transition_matrix(hmm)
    @unpack logL, logb, α, α_next = f

    obs_logdensities!(logb, hmm, obs_seq[1])
    check_right_finite(logb)
    logm = maximum(logb)
    α .= p .* exp.(logb .- logm)
    c = inv(sum(α))
    α .*= c
    check_finite(α)
    logL[] = -log(c) + logm
    for t in 1:(T - 1)
        obs_logdensities!(logb, hmm, obs_seq[t + 1])
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

function forward!(
    fs::Vector{<:ForwardStorage},
    hmm::AbstractHMM,
    obs_seqs::Vector{<:Vector},
    nb_seqs::Integer,
)
    check_nb_seqs(obs_seqs, nb_seqs)
    @threads for k in eachindex(fs, obs_seqs)
        forward!(fs[k], hmm, obs_seqs[k])
    end
    return nothing
end

"""
    forward(hmm, obs_seq)
    forward(hmm, obs_seqs, nb_seqs)

Run the forward algorithm to infer the current state of an HMM.
    
When applied on a single sequence, this function returns a tuple `(α, logL)` where

- `α[i]` is the posterior probability of state `i` at the end of the sequence
- `logL` is the loglikelihood of the sequence

When applied on multiple sequences, this function returns a vector of tuples.
"""
function forward(hmm::AbstractHMM, obs_seqs::Vector{<:Vector}, nb_seqs::Integer)
    check_nb_seqs(obs_seqs, nb_seqs)
    fs = [initialize_forward(hmm, obs_seqs[k]) for k in eachindex(obs_seqs)]
    forward!(fs, hmm, obs_seqs, nb_seqs)
    return [(f.α, f.logL[]) for f in fs]
end

function forward(hmm::AbstractHMM, obs_seq::Vector)
    return only(forward(hmm, [obs_seq], 1))
end

"""
    logdensityof(hmm, obs_seq)
    logdensityof(hmm, obs_seqs, nb_seqs)

Run the forward algorithm to compute the posterior loglikelihood of observations for an HMM.

Whether it is applied on one or multiple sequences, this function returns a number.
"""
function DensityInterface.logdensityof(
    hmm::AbstractHMM, obs_seqs::Vector{<:Vector}, nb_seqs::Integer
)
    logαs_and_logLs = forward(hmm, obs_seqs, nb_seqs)
    return sum(last, logαs_and_logLs)
end

function DensityInterface.logdensityof(hmm::AbstractHMM, obs_seq::Vector)
    return logdensityof(hmm, [obs_seq], 1)
end
