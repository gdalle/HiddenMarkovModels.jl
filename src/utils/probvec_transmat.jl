function valid_prob_vec(p::AbstractVector; atol=1e-2)
    return (minimum(p) >= 0) && isapprox(sum(p), 1; atol=atol)
end

function is_square(A::AbstractMatrix)
    return size(A, 1) == size(A, 2)
end

function valid_trans_mat(A::AbstractMatrix; atol=1e-2)
    if !is_square(A)
        return false
    else
        for row in eachrow(A)
            if !valid_prob_vec(row; atol=atol)
                return false
            end
        end
        return true
    end
end

"""
    rand_prob_vec(N)
    rand_prob_vec(rng, N)

Generate a random probability distribution of size `N`.
"""
function rand_prob_vec(rng::AbstractRNG, N)
    p = rand(rng, N)
    sum_to_one!(p)
    return p
end

"""
    rand_trans_mat(N)
    rand_trans_mat(rng, N)

Generate a random transition matrix of size `(N, N)`.
"""
function rand_trans_mat(rng::AbstractRNG, N)
    A = rand(rng, N, N)
    foreach(sum_to_one!, eachrow(A))
    return A
end

rand_prob_vec(N) = rand_prob_vec(default_rng(), N)
rand_trans_mat(N) = rand_trans_mat(default_rng(), N)
