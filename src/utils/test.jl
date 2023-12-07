function similar_hmms(
    hmm1::AbstractHMM, hmm2::AbstractHMM; control_seq=[nothing], atol=1e-5, test_init=false
)
    if test_init
        init1 = initialization(hmm1)
        init2 = initialization(hmm2)
        if maximum(abs, init1 - init2) > atol
            @warn "Error in initialization"
            display(init1)
            display(init1)
            return false
        end
    end

    for control in control_seq
        trans1 = transition_matrix(hmm1, control)
        trans2 = transition_matrix(hmm2, control)
        if maximum(abs, trans1 - trans2) > atol
            @warn "Error in transition matrix" control
            display(trans1)
            display(trans2)
            return false
        end
    end

    for control in control_seq
        dists1 = obs_distributions(hmm1, control)
        dists2 = obs_distributions(hmm2, control)
        for (dist1, dist2) in zip(dists1, dists2)
            for field in fieldnames(typeof(dist1))
                if startswith(string(field), "log")
                    continue
                end
                x1 = getfield(dist1, field)
                x2 = getfield(dist2, field)
                if maximum(abs, x1 - x2) > atol
                    @warn "Error in observation distribution" control field
                    display(x1)
                    display(x2)
                    return false
                end
            end
        end
    end

    return true
end

function coherent_algorithms(
    rng::AbstractRNG,
    hmm::AbstractHMM,
    hmm_guess::Union{Nothing,AbstractHMM}=nothing;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    kwargs...,
)
    simulations = map(eachindex(seq_ends)) do k
        t1, t2 = seq_limits(seq_ends, k)
        rand(rng, hmm, control_seq[t1:t2])
    end

    state_seqs = [sim.state_seq for sim in simulations]
    obs_seqs = [sim.obs_seq for sim in simulations]

    state_seq = reduce(vcat, state_seqs)
    obs_seq = reduce(vcat, obs_seqs)

    logL = logdensityof(hmm, obs_seq; control_seq, seq_ends)
    logL_joint = logdensityof(hmm, obs_seq, state_seq; control_seq, seq_ends)

    q, logL_viterbi = viterbi(hmm, obs_seq; control_seq, seq_ends)
    if logL_viterbi < logL_joint
        @warn "Viterbi joint loglikelihood is not maximal"
        return false
    elseif !(logL_viterbi ≈ logdensityof(hmm, obs_seq, q; control_seq, seq_ends))
        @warn "Viterbi joint loglikelihood incoherent with best state sequence"
        return false
    end

    α, logL_forward = forward(hmm, obs_seq; control_seq, seq_ends)
    if !(logL_forward ≈ logL)
        @warn "Forward loglikelihood incoherent with logdensityof"
        return false
    end

    γ, logL_forward_backward = forward_backward(hmm, obs_seq; control_seq, seq_ends)
    if !(logL_forward_backward ≈ logL)
        @warn "Forward-backward loglikelihood incoherent with logdensityof"
        return false
    elseif !all(α[:, seq_ends[k]] ≈ γ[:, seq_ends[k]] for k in eachindex(seq_ends))
        @warn "Forward filtered marginals incoherent with forward-backward marginals"
        return false
    end

    if !isnothing(hmm_guess)
        hmm_est, logL_evolution = baum_welch(hmm_guess, obs_seq; control_seq, seq_ends)
        if any(<(0), diff(logL_evolution))
            @warn "Baum-Welch loglikelihood not increasing"
            return false
        elseif !similar_hmms(hmm, hmm_est; control_seq=control_seq[1:2], kwargs...)
            @warn "Baum-Welch estimation imprecise"
            return false
        end
    end

    return true
end
