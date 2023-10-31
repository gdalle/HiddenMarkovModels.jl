function baum_welch!(
    hmm::AbstractHMM, obs_seqs; atol, max_iterations, check_loglikelihood_increasing
)
    # Pre-allocate nearly all necessary memory
    logB = loglikelihoods(hmm, obs_seqs[1])
    fb = forward_backward(hmm, logB)

    logBs = Vector{typeof(logB)}(undef, length(obs_seqs))
    fbs = Vector{typeof(fb)}(undef, length(obs_seqs))
    logBs[1], fbs[1] = logB, fb
    @threads for k in eachindex(obs_seqs)
        if k > 1
            logBs[k] = loglikelihoods(hmm, obs_seqs[k])
            fbs[k] = forward_backward(hmm, logBs[k])
        end
    end

    logL = loglikelihood(fbs)
    logL_evolution = [logL]

    for iteration in 1:max_iterations
        # E step
        if iteration > 1
            @threads for k in eachindex(obs_seqs, logBs, fbs)
                loglikelihoods!(logBs[k], hmm, obs_seqs[k])
                forward_backward!(fbs[k], hmm, logBs[k])
            end
            logL = loglikelihood(fbs)
            push!(logL_evolution, logL)
        end

        # M step
        fit!(hmm, obs_seqs, fbs)

        #  Stopping criterion
        if iteration > 1
            progress = logL_evolution[end] - logL_evolution[end - 1]
            if check_loglikelihood_increasing && progress < 0
                error("Loglikelihood decreased in Baum-Welch")
            elseif progress < atol
                break
            end
        end
    end

    return logL_evolution
end

"""
    baum_welch(
        hmm_init, obs_seq;
        atol, max_iterations, check_loglikelihood_increasing
    )

Apply the Baum-Welch algorithm to estimate the parameters of an HMM starting from `hmm_init`.

Return a tuple `(hmm_est, logL_evolution)`.

# Keyword arguments

- `atol`: Minimum loglikelihood increase at an iteration of the algorithm (otherwise the algorithm is deemed to have converged)
- `max_iterations`: Maximum number of iterations of the algorithm
- `check_loglikelihood_increasing`: Whether to throw an error if the loglikelihood decreases
"""
function baum_welch(
    hmm_init::AbstractHMM,
    obs_seq;
    atol=1e-5,
    max_iterations=100,
    check_loglikelihood_increasing=true,
)
    hmm = deepcopy(hmm_init)
    logL_evolution = baum_welch!(
        hmm, [obs_seq]; atol, max_iterations, check_loglikelihood_increasing
    )
    return hmm, logL_evolution
end

"""
    baum_welch(
        hmm_init, obs_seqs, nb_seqs;
        atol, max_iterations, check_loglikelihood_increasing
    )

Apply the Baum-Welch algorithm to estimate the parameters of an HMM starting from `hmm_init`, based on `nb_seqs` observation sequences.

Return a tuple `(hmm_est, logL_evolution)`.

!!! warning "Multithreading"
    This function is parallelized across sequences.

# Keyword arguments

- `atol`: Minimum loglikelihood increase at an iteration of the algorithm (otherwise the algorithm is deemed to have converged)
- `max_iterations`: Maximum number of iterations of the algorithm
- `check_loglikelihood_increasing`: Whether to throw an error if the loglikelihood decreases
"""
function baum_welch(
    hmm_init::AbstractHMM,
    obs_seqs,
    nb_seqs::Integer;
    atol=1e-5,
    max_iterations=100,
    check_loglikelihood_increasing=true,
)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    hmm = deepcopy(hmm_init)
    logL_evolution = baum_welch!(
        hmm, obs_seqs; atol, max_iterations, check_loglikelihood_increasing
    )
    return hmm, logL_evolution
end
