"""
    baum_welch(hmm::HMM, obs_seqs; max_iterations, tol)

Apply the Baum-Welch algorithm on multiple observation sequences, starting from an initial estimate `hmm`.
"""
function baum_welch!(hmm::HMM, obs_seqs; max_iterations=100, tol=1e-5)
    θ = nothing
    N = nb_states(hmm, θ)

    p = initial_distribution(hmm, θ)
    A = transition_matrix(hmm, θ)

    Bs = [likelihoods(hmm, θ, obs_seq) for obs_seq in obs_seqs]
    forbacks = [initialize_forward_backward(p, A, B) for B in Bs]
    logL_evolution = Float64[]

    for iteration in 1:max_iterations
        logL = 0.0
        for k in eachindex(obs_seqs, Bs, forbacks)
            obs_seq, B, forback = obs_seqs[k], Bs[k], forbacks[k]
            likelihoods!(B, hmm, θ, obs_seq)
            logL += forward_backward!(forback, p, A, B)
        end
        push!(logL_evolution, logL)

        update_initial_distribution!(hmm, forbacks)
        update_transition_matrix!(hmm, forbacks)
        update_emission_distributions!(hmm, forbacks, obs_seqs)

        if (iteration > 1) && (logL_evolution[end] - logL_evolution[end - 1] < tol)
            break
        end
    end
    return logL_evolution
end

function update_initial_distribution!(hmm, forbacks)
    p = hmm.initial_distribution
    p .= zero(eltype(p))
    @views for k in eachindex(forbacks)
        p .+= forbacks[k].γ[:, 1]
    end
    p ./= sum(p)
    check_nan(p)
    return nothing
end

function update_transition_matrix!(hmm, forbacks)
    A = hmm.transition_matrix
    A .= zero(eltype(A))
    for k in eachindex(forbacks)
        A .+= dropdims(sum(forbacks[k].ξ; dims=3); dims=3)
    end
    A ./= sum(A; dims=2)
    check_nan(A)
    return nothing
end

function update_emission_distributions!(
    hmm::HMM{T1,T2,D}, forbacks, obs_seqs
) where {T1,T2,D}
    em = hmm.emission_distributions
    N = length(em)
    xs = (obs_seqs[k] for k in eachindex(obs_seqs, forbacks))
    @views for i in 1:N
        ws = (forbacks[k].γ[i, :] for k in eachindex(obs_seqs, forbacks))
        em[i] = fit_mle_from_multiple_sequences(D, xs, ws)
    end
end
