function baum_welch!(
    hmm::AbstractHMM, obs_seqs; atol, max_iterations, check_loglikelihood_increasing
)
    # Pre-allocate nearly all necessary memory
    fb = initialize_forward_backward(hmm, obs_seqs[1])
    fbs = Vector{typeof(fb)}(undef, length(obs_seqs))
    @threads for k in eachindex(obs_seqs, fbs)
        fbs[k] = initialize_forward_backward(hmm, obs_seqs[k])
    end

    init_count, trans_count = initialize_states_stats(fbs)
    state_marginals_concat = initialize_observations_stats(fbs)
    obs_seqs_concat = reduce(vcat, obs_seqs)
    logL_evolution = eltype(fbs[1])[]

    for iteration in 1:max_iterations
        # E step
        @threads for k in eachindex(obs_seqs, fbs)
            forward_backward!(fbs[k], hmm, obs_seqs[k])
        end
        logL = loglikelihood(fbs)
        push!(logL_evolution, logL)

        # M step
        update_states_stats!(init_count, trans_count, fbs)
        update_observations_stats!(state_marginals_concat, fbs)
        fit!(hmm, init_count, trans_count, obs_seqs_concat, state_marginals_concat)

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
