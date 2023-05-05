"""
    baum_welch(hmm_init::HMM, obs_seqs; max_iterations, tol)

Apply the Baum-Welch algorithm on multiple observation sequences, starting from an initial [`HMM`](@ref) `hmm_init`.
"""
function baum_welch(hmm_init::H, obs_seqs; max_iterations=100, tol=1e-5) where {H<:HMM}
    θ = nothing
    N = nb_states(hmm_init, θ)
    hmm = hmm_init

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

        new_p = estimate_initial_distribution(forbacks)
        new_A = estimate_transition_matrix(forbacks)
        new_em = [estimate_emission_distribution(H, forbacks, obs_seqs, i) for i in 1:N]
        hmm = H(new_p, new_A, new_em)

        if (iteration > 1) && (logL_evolution[end] - logL_evolution[end - 1] < tol)
            break
        end
    end
    return hmm, logL_evolution
end

function estimate_initial_distribution(forbacks)
    @views p = Vector(reduce(+, forbacks[k].γ[:, 1] for k in eachindex(forbacks)))
    p ./= sum(p)
    check_nan(p)
    return p
end

function estimate_transition_matrix(forbacks)
    A = reduce(+, dropdims(sum(forbacks[k].ξ; dims=3); dims=3) for k in eachindex(forbacks))
    A ./= sum(A; dims=2)
    check_nan(A)
    return A
end

function estimate_emission_distribution(
    ::Type{HMM{T1,T2,D}}, forbacks, obs_seqs, i
) where {T1,T2,D}
    xs = (obs_seqs[k] for k in eachindex(obs_seqs, forbacks))
    ws = (forbacks[k].γ[i, :] for k in eachindex(obs_seqs, forbacks))
    return fit_mle_from_multiple_sequences(D, xs, ws)
end
