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

function check_dists(d)
    for i in eachindex(d)
        if DensityKind(d[i]) == NoDensity()
            throw(ArgumentError("Observation is not a density"))
        end
    end
end

"""
    check_hmm(hmm::AbstractHMM)

Verify that `hmm` satisfies basic assumptions.
"""
function check_hmm(hmm::AbstractHMM)
    p = initialization(hmm)
    A = transition_matrix(hmm)
    d = obs_distributions(hmm)
    if !all(==(length(hmm)), (length(p), size(A, 1), size(A, 2), length(d)))
        throw(DimensionMismatch("Incoherent sizes"))
    end
    check_prob_vec(p)
    check_trans_mat(A)
    check_dists(d)
    return nothing
end

function check_lengths(obs_seqs::Vector{<:Vector}, nb_seqs::Integer)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
end
