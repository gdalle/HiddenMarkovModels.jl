function similar_hmms(
    hmm1::AbstractHMM, hmm2::AbstractHMM; control_seq=[nothing], atol=1e-5, test_init=false
)
    if test_init
        init1 = initialization(hmm1)
        init2 = initialization(hmm2)
        if maximum(abs, init1 - init2) > atol
            @warn "Error in initialization" init1 init2
            return false
        end
    end

    for control in control_seq
        trans1 = transition_matrix(hmm1, control)
        trans2 = transition_matrix(hmm2, control)
        if maximum(abs, trans1 - trans2) > atol
            @warn "Error in transition matrix" control trans1 trans2
            return false
        end
    end

    for control in control_seq
        dists1 = obs_distributions(hmm1, control)
        dists2 = obs_distributions(hmm2, control)
        for (dist1, dist2) in zip(dists1, dists2)
            for field in fieldnames(typeof(dist1))
                x1 = getfield(dist1, field)
                x2 = getfield(dist2, field)
                if maximum(abs, x1 - x2) > atol
                    @warn "Error in observation distribution" control field x1 x2
                    return false
                end
            end
        end
    end

    return true
end
