function reestimate!(::Tr, fbs::MultiForwardBackwardStorage) where {Tr<:AbstractTransitions}
    return error(
        "`$Tr` needs to implement `reestimate!` for the Baum-Welch algorithm to work."
    )
end

function reestimate!(
    ::Em, fbs::MultiForwardBackwardStorage, obs_seqs::Vector{<:Vector}
) where {Em<:AbstractEmissions}
    return error(
        "`$Em` needs to implement `reestimate!` for the Baum-Welch algorithm to work."
    )
end

"""
    baum_welch(hmm::HMM, obs_seqs; max_iterations, rtol)

Apply the Baum-Welch algorithm on multiple observation sequences, starting from an initial estimate `hmm`.
"""
function baum_welch!(hmm::HMM, obs_seqs::Vector; max_iterations=100, rtol=1e-3)
    p = initial_distribution(hmm)
    A = transition_matrix(hmm)
    Bs = [likelihoods(hmm, obs_seq) for obs_seq in obs_seqs]

    # Pre-allocate all necessary memory
    logL_by_seq = fill(NaN, length(obs_seqs))
    logL_evolution = fill(NaN, max_iterations)
    fbs = [initialize_forward_backward(p, A, B) for B in Bs]
    p_count, A_count = initialize_transitions_stats(fbs)
    γ_concat = initialize_emissions_stats(fbs)
    obs_seqs_concat = reduce(vcat, obs_seqs)

    for iteration in 1:max_iterations
        # E step by sequence
        @threads for k in eachindex(obs_seqs, Bs, fbs)
            obs_seq, B, fb = obs_seqs[k], Bs[k], fbs[k]
            likelihoods!(B, hmm, obs_seq)
            logL_by_seq[k] = float(forward_backward!(fb, p, A, B))
        end
        logL = sum(logL_by_seq)
        logL_evolution[iteration] = logL

        #  Stopping criterion
        logL_prev = iteration > 1 ? logL_evolution[iteration - 1] : NaN
        if (logL - logL_prev) / abs(logL_prev) < rtol
            max_iterations = iteration
            break
        end

        # M step
        update_transitions_stats!(p_count, A_count, fbs)
        update_emissions_stats!(γ_concat, fbs)
        reestimate!(hmm.transitions, p_count, A_count)
        reestimate!(hmm.emissions, obs_seqs_concat, γ_concat)
    end

    return logL_evolution[1:max_iterations]
end
