function check_finite(a)
    if !all(isfinite, mynonzeros(a))
        throw(OverflowError("Some values are infinite or NaN"))
    end
end

function check_right_finite(a)
    if !all(<(typemax(eltype(a))), mynonzeros(a))
        throw(OverflowError("Some values are positive infinite or NaN"))
    end
end

function check_no_nan(a)
    if any(isnan, mynonzeros(a))
        throw(OverflowError("Some values are NaN"))
    end
end

function check_positive(a)
    if !all(>(zero(eltype(a))), mynonzeros(a))
        throw(OverflowError("Some values are not positive"))
    end
end

function check_nonnegative(a)
    if any(<(zero(eltype(a))), mynonzeros(a))
        throw(OverflowError("Some values are negative"))
    end
end

function check_prob_vec(p::AbstractVector)
    check_finite(p)
    if !valid_prob_vec(p)
        throw(ArgumentError("Invalid probability distribution."))
    end
end

function check_trans_mat(A::AbstractMatrix)
    check_finite(A)
    if !valid_trans_mat(A)
        throw(ArgumentError("Invalid transition matrix."))
    end
end

function check_dists(d::AbstractVector)
    for i in eachindex(d)
        if DensityKind(d[i]) == NoDensity()
            throw(
                ArgumentError(
                    "Invalid observation distributions (do not satisfy DensityInterface.jl)"
                ),
            )
        end
    end
    return true
end

function check_hmm_sizes(p::AbstractVector, A::AbstractMatrix, d::AbstractVector)
    if !(size(A) == (length(p), length(p)) == (length(d), length(d)))
        throw(
            DimensionMismatch(
                "Initialization, transition matrix and observation distributions have incompatible sizes.",
            ),
        )
    end
end

function check_hmm(hmm::AbstractHMM)
    p = initialization(hmm)
    A = transition_matrix(hmm, 1)
    d = obs_distributions(hmm, 1)
    check_hmm_sizes(p, A, d)
    check_prob_vec(p)
    check_trans_mat(A)
    check_dists(d)
    return nothing
end

function check_nb_seqs(obs_seqs::Vector{<:Vector}, nb_seqs::Integer)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("Incoherent sizes provided: `nb_seqs != length(obs_seqs)`"))
    end
end
