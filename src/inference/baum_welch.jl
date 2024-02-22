function baum_welch_has_converged(
    logL_evolution::Vector; atol::Real, loglikelihood_increasing::Bool
)
    if length(logL_evolution) >= 2
        logL, logL_prev = logL_evolution[end], logL_evolution[end - 1]
        progress = logL - logL_prev
        if loglikelihood_increasing && progress < min(0, -atol)
            error("Loglikelihood decreased in Baum-Welch")
        elseif progress < atol
            return true
        end
    end
    return false
end

"""
$(SIGNATURES)
"""
function baum_welch!(
    fb_storage::ForwardBackwardStorage,
    logL_evolution::Vector,
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVector{Int},
    atol::Real,
    max_iterations::Integer,
    loglikelihood_increasing::Bool,
)
    for _ in 1:max_iterations
        forward_backward!(fb_storage, hmm, obs_seq, control_seq; seq_ends)
        push!(logL_evolution, logdensityof(hmm) + sum(fb_storage.logL))
        fit!(hmm, fb_storage, obs_seq, control_seq; seq_ends)
        if baum_welch_has_converged(logL_evolution; atol, loglikelihood_increasing)
            break
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the Baum-Welch algorithm to estimate the parameters of an HMM on `obs_seq`, starting from `hmm_guess`.

Return a tuple `(hmm_est, loglikelihood_evolution)` where `hmm_est` is the estimated HMM and `loglikelihood_evolution` is a vector of loglikelihood values, one per iteration of the algorithm.

# Keyword arguments

- `atol`: minimum loglikelihood increase at an iteration of the algorithm (otherwise the algorithm is deemed to have converged)
- `max_iterations`: maximum number of iterations of the algorithm
- `loglikelihood_increasing`: whether to throw an error if the loglikelihood decreases
"""
function baum_welch(
    hmm_guess::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVector{Int}=Fill(length(obs_seq), 1),
    atol=1e-5,
    max_iterations=100,
    loglikelihood_increasing=true,
)
    hmm = deepcopy(hmm_guess)
    fb_storage = initialize_forward_backward(hmm, obs_seq, control_seq; seq_ends)
    logL_evolution = eltype(fb_storage)[]
    sizehint!(logL_evolution, max_iterations)
    baum_welch!(
        fb_storage,
        logL_evolution,
        hmm,
        obs_seq,
        control_seq;
        seq_ends,
        atol,
        max_iterations,
        loglikelihood_increasing=false,
    )
    return hmm, logL_evolution
end

## Fallback

function StatsAPI.fit!(
    hmm::AbstractHMM,
    fb_storage::ForwardBackwardStorage,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVector{Int},
)
    return fit!(hmm, fb_storage, obs_seq; seq_ends)
end
