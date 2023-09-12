function is_square(A::AbstractMatrix)
    return size(A, 1) == size(A, 2)
end

function is_trans_mat(A::AbstractMatrix; atol=1e-2)
    if !is_square(A)
        return false
    else
        for row in eachrow(A)
            if !is_prob_vec(row; atol=atol)
                return false
            end
        end
        return true
    end
end

function rand_trans_mat(rng::AbstractRNG, N)
    A = rand(rng, N, N)
    foreach(sum_to_one!, eachrow(A))
    return A
end

rand_trans_mat(N) = rand_trans_mat(default_rng(), N)
