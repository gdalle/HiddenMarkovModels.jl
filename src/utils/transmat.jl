"""
    is_square(A)

Check if `A` is a square matrix.
"""
function is_square(A::AbstractMatrix)
    return size(A, 1) == size(A, 2)
end

"""
    is_trans_mat(A; rtol)

Check if `A` is a transition matrix.
"""
function is_trans_mat(A::AbstractMatrix; rtol=1e-2)
    if !is_square(A)
        return false
    else
        for row in eachrow(A)
            if !is_prob_vec(row; rtol=rtol)
                return false
            end
        end
        return true
    end
end

function check_trans_mat(A::AbstractMatrix)
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

"""
    rand_trans_mat(rng, N)
    rand_trans_mat(N)

Generate a random transition matrix of size `N × N`.
"""
function rand_trans_mat(rng::AbstractRNG, N)
    A = rand(rng, N, N)
    foreach(sum_to_one!, eachrow(A))
    return A
end

rand_trans_mat(N) = rand_trans_mat(GLOBAL_RNG, N)
