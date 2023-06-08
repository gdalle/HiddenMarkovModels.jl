function baum_welch!(hmm::HMM, obs_seqs; max_iterations, rtol)
    # Pre-allocate nearly all necessary memory
    logBs = loglikelihoods.(Ref(hmm.obs_process), obs_seqs)
    fbs = [initialize_forward_backward(hmm.state_process, logB) for logB in logBs]
    p_count, A_count = initialize_states_stats(fbs)
    γ_concat = initialize_observations_stats(fbs)
    obs_seqs_concat = reduce(vcat, obs_seqs)

    # First E step
    @threads for k in eachindex(obs_seqs, logBs, fbs)
        loglikelihoods!(logBs[k], hmm.obs_process, obs_seqs[k])
        forward_backward!(fbs[k], hmm.state_process, logBs[k])
    end
    logL = loglikelihood(fbs)
    logL_evolution = [logL]

    for iteration in 1:max_iterations
        # E step
        if iteration > 1
            @threads for k in eachindex(obs_seqs, logBs, fbs)
                loglikelihoods!(logBs[k], hmm.obs_process, obs_seqs[k])
                forward_backward!(fbs[k], hmm.state_process, logBs[k])
            end
            logL = loglikelihood(fbs)
            push!(logL_evolution, logL)
        end

        # M step
        update_states_stats!(p_count, A_count, fbs)
        update_observations_stats!(γ_concat, fbs)
        reestimate!(hmm.state_process, p_count, A_count)
        reestimate!(hmm.obs_process, obs_seqs_concat, γ_concat)

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
    baum_welch(hmm_init, obs_seqs; max_iterations, rtol)

Apply the Baum-Welch algorithm to estimate the parameters of an HMM on multiple observation sequences, and return a tuple `(hmm, logL_evolution)`.
"""
function baum_welch(hmm_init::HMM, obs_seqs; max_iterations=100, rtol=1e-3)
    hmm = copy(hmm_init)
    logL_evolution = baum_welch!(hmm, obs_seqs; max_iterations, rtol)
    return hmm, logL_evolution
end
