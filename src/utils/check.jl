function check_no_nan(a)
    if any(isnan, a)
        throw(OverflowError("Some values are NaN"))
    end
end

function check_no_inf(a)
    if any(isequal(typemax(eltype(a))), a)
        throw(OverflowError("Some values are infinite"))
    end
end

function check_positive(a)
    if any(!>(zero(eltype(a))), a)
        throw(OverflowError("Some values are not positive"))
    end
end

function check_prob_vec(p::AbstractVector)
    check_no_nan(p)
    if !is_prob_vec(p)
        throw(ArgumentError("Invalid probability distribution."))
    end
end

function check_trans_mat(A::AbstractMatrix)
    check_no_nan(A)
    if !is_trans_mat(A)
        throw(ArgumentError("Invalid transition matrix."))
    end
end

function check_coherent_sizes(p::AbstractVector, A::AbstractMatrix)
    if size(A) != (length(p), length(p))
        throw(
            DimensionMismatch(
                "Probability distribution and transition matrix are incompatible."
            ),
        )
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
    init = initialization(mc)
    trans = transition_matrix(mc)
    if !(length(init) == size(trans, 1) == size(trans, 2))
        throw(DimensionMismatch("Incoherent sizes"))
    end
    check_prob_vec(init)
    check_trans_mat(trans)
    return nothing
end

function check_hmm(hmm::AbstractHMM)
    mc = MarkovChain(hmm)
    dists = [obs_distribution(hmm, i) for i in 1:length(hmm)]
    if length(mc) != length(dists)
        throw(DimensionMismatch("Incoherent sizes"))
    end
    check_mc(mc)
    check_dists(dists)
    return nothing
end
