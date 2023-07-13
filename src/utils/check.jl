function check_no_nan(a)
    if any(isnan, a)
        throw(OverflowError("Some values are NaN"))
    end
end

function check_positive(a)
    if any(!>(zero(eltype(a))), a)
        throw(OverflowError("Some values are not positive"))
    end
end

function check_dists(dists)
    for i in eachindex(dists)
        if DensityKind(dists[i]) == NoDensity()
            throw(ArgumentError("Observation is not a density"))
        end
    end
end

function check_mc(mc::MarkovChain)
    init = initial_distribution(mc)
    trans = transition_matrix(mc)
    if !(length(init) == size(trans, 1) == size(trans, 2))
        throw(DimensionMismatch("Incoherent sizes"))
    end
    check_prob_vec(init)
    check_trans_mat(trans)
    return nothing
end

function check_hmm(hmm::AbstractHMM)
    mc = MarkovChain(initial_distribution(hmm), transition_matrix(hmm))
    dists = [obs_distribution(hmm, i) for i in 1:length(hmm)]
    if length(mc) != length(dists)
        throw(DimensionMismatch("Incoherent sizes"))
    end
    check_mc(mc)
    check_dists(dists)
    return nothing
end
