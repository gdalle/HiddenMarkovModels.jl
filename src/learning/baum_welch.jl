function baum_welch!(hmm::AbstractHMM, obs_seqs; max_iterations, rtol)
    # Pre-allocate nearly all necessary memory
    logB = loglikelihoods(hmm, obs_seqs[1])
    fb = initialize_forward_backward(hmm, logB)

    logBs = Vector{typeof(logB)}(undef, length(obs_seqs))
    fbs = Vector{typeof(fb)}(undef, length(obs_seqs))
    @threads for k in eachindex(obs_seqs)
        logBs[k] = loglikelihoods(hmm, obs_seqs[k])
        fbs[k] = forward_backward_from_loglikelihoods(hmm, logBs[k])
    end

    init_count, trans_count = initialize_states_stats(fbs)
    state_marginals_concat = initialize_observations_stats(fbs)
    obs_seqs_concat = reduce(vcat, obs_seqs)
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
        update_states_stats!(init_count, trans_count, fbs)
        update_observations_stats!(state_marginals_concat, fbs)
        fit!(hmm, init_count, trans_count, obs_seqs_concat, state_marginals_concat)

        #  Stopping criterion
        if iteration > 1
            progress = (
                (logL_evolution[end] - logL_evolution[end - 1]) /
                abs(logL_evolution[end - 1])
            )
            if progress < -eps(progress)
                error("Loglikelihood decreased in Baum-Welch")
            elseif progress < rtol
                break
            end
        end
    end

    return logL_evolution
end

"""
    baum_welch(hmm_init, obs_seq; max_iterations, rtol)

Apply the Baum-Welch algorithm to estimate the parameters of an HMM and return a tuple `(hmm, logL_evolution)`.

The procedure is based on a single observation sequence and initialized with `hmm_init`.
"""
function baum_welch(hmm_init::AbstractHMM, obs_seq; max_iterations=100, rtol=1e-3)
    hmm = deepcopy(hmm_init)
    logL_evolution = baum_welch!(hmm, [obs_seq]; max_iterations, rtol)
    return hmm, logL_evolution
end

"""
    baum_welch(hmm_init, obs_seqs, nb_seqs; max_iterations, rtol)

Apply the Baum-Welch algorithm to estimate the parameters of an HMM and return a tuple `(hmm, logL_evolution)`.

The procedure is based on multiple observation sequences and initialized with `hmm_init`.
"""
function baum_welch(
    hmm_init::AbstractHMM, obs_seqs, nb_seqs::Integer; max_iterations=100, rtol=1e-3
)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    hmm = deepcopy(hmm_init)
    logL_evolution = baum_welch!(hmm, obs_seqs; max_iterations, rtol)
    return hmm, logL_evolution
end
