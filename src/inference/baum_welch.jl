function reestimate!(
    ::Tr, forbacks::Vector{<:ForwardBackward}
) where {Tr<:AbstractTransitions}
    return error(
        "`$Tr` needs to implement `reestimate!` for the Baum-Welch algorithm to work."
    )
end

function reestimate!(
    ::Em, forbacks::Vector{<:ForwardBackward}, obs_seqs::Vector{<:Vector}
) where {Em<:AbstractEmissions}
    return error(
        "`$Em` needs to implement `reestimate!` for the Baum-Welch algorithm to work."
    )
end

"""
    baum_welch(hmm::HMM, obs_seqs; max_iterations, rtol)

Apply the Baum-Welch algorithm on multiple observation sequences, starting from an initial estimate `hmm`.
"""
function baum_welch!(hmm::HMM, obs_seqs; max_iterations=100, rtol=1e-3)
    p = initial_distribution(hmm)
    A = transition_matrix(hmm)
    Bs = [likelihoods(hmm, obs_seq) for obs_seq in obs_seqs]
    forbacks = [initialize_forward_backward(p, A, B) for B in Bs]
    logL_evolution = Float64[]

    for iteration in 1:max_iterations
        logL = 0.0
        for k in eachindex(obs_seqs, Bs, forbacks)
            obs_seq, B, forback = obs_seqs[k], Bs[k], forbacks[k]
            likelihoods!(B, hmm, obs_seq)
            logL += forward_backward!(forback, p, A, B)
        end
        push!(logL_evolution, logL)

        reestimate!(hmm.transitions, forbacks)
        reestimate!(hmm.emissions, forbacks, obs_seqs)

        if iteration > 1
            logL_prev = logL_evolution[end - 1]
            if (logL - logL_prev) / abs(logL_prev) < rtol
                break
            end
        end
    end
    return logL_evolution
end
