function baum_welch_has_converged(
    logL_evolution::Vector; atol::Real, loglikelihood_increasing::Bool
)
    if length(logL_evolution) >= 2
        logL, logL_prev = logL_evolution[end], logL_evolution[end - 1]
        progress = logL - logL_prev
        if loglikelihood_increasing && progress < 0
            error("Loglikelihood decreased in Baum-Welch")
        elseif progress < atol
            return true
        end
    end
    return false
end

function baum_welch!(
    fb::ForwardBackwardStorage,
    logL_evolution::Vector,
    hmm::AbstractHMM,
    obs_seq::Vector;
    atol::Real,
    max_iterations::Integer,
    loglikelihood_increasing::Bool,
)
    for _ in 1:max_iterations
        forward_backward!(fb, hmm, obs_seq)
        push!(logL_evolution, fb.logL[])
        fit!(hmm, (fb,), (obs_seq,), fb, obs_seq)
        if baum_welch_has_converged(logL_evolution; atol, loglikelihood_increasing)
            break
        end
    end
    return nothing
end

function baum_welch!(
    fbs::Vector{<:ForwardBackwardStorage},
    fb_concat::ForwardBackwardStorage,
    logL_evolution::Vector,
    hmm::AbstractHMM,
    obs_seqs::Vector{<:Vector},
    obs_seqs_concat::Vector;
    atol::Real,
    max_iterations::Integer,
    loglikelihood_increasing::Bool,
)
    for _ in 1:max_iterations
        @threads for k in eachindex(obs_seqs, fbs)
            forward_backward!(fbs[k], hmm, obs_seqs[k])
        end
        push!(logL_evolution, sum(fb.logL[] for fb in fbs))
        fit!(hmm, fbs, obs_seqs, fb_concat, obs_seqs_concat)
        if baum_welch_has_converged(logL_evolution; atol, loglikelihood_increasing)
            break
        end
    end
    return nothing
end

"""
    baum_welch(hmm_init, obs_seq; kwargs...)
    baum_welch(hmm_init, obs_seqs, nb_seqs; kwargs...)

Apply the Baum-Welch algorithm to estimate the parameters of an HMM starting from `hmm_init`, based on one or several observation sequences.

Return a tuple `(hmm_est, logL_evolution)`.

# Keyword arguments

- `atol`: minimum loglikelihood increase at an iteration of the algorithm (otherwise the algorithm is deemed to have converged)
- `max_iterations`: maximum number of iterations of the algorithm
- `loglikelihood_increasing`: whether to throw an error if the loglikelihood decreases
"""
function baum_welch(
    hmm_init::AbstractHMM,
    obs_seq::Vector;
    atol=1e-5,
    max_iterations=100,
    loglikelihood_increasing=true,
)
    hmm = deepcopy(hmm_init)
    fb = initialize_forward_backward(hmm, obs_seq)
    R = eltype(hmm, obs_seq[1])
    logL_evolution = R[]
    sizehint!(logL_evolution, max_iterations)
    baum_welch!(
        fb, logL_evolution, hmm, obs_seq; atol, max_iterations, loglikelihood_increasing
    )
    return hmm, logL_evolution
end

function baum_welch(
    hmm_init::AbstractHMM,
    obs_seqs::Vector{<:Vector},
    nb_seqs::Integer;
    atol=1e-5,
    max_iterations=100,
    loglikelihood_increasing=true,
)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    hmm = deepcopy(hmm_init)
    limits = vcat(0, cumsum(length.(obs_seqs)))
    obs_seqs_concat = reduce(vcat, obs_seqs)
    fb_concat = initialize_forward_backward(hmm, obs_seqs_concat)
    fbs = [view(fb_concat, (limits[k] + 1):limits[k + 1]) for k in eachindex(obs_seqs)]
    R = eltype(hmm, obs_seqs[1][1])
    logL_evolution = R[]
    sizehint!(logL_evolution, max_iterations)
    baum_welch!(
        fbs,
        fb_concat,
        logL_evolution,
        hmm,
        obs_seqs,
        obs_seqs_concat;
        atol,
        max_iterations,
        loglikelihood_increasing,
    )
    return hmm, logL_evolution
end
