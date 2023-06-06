"""
    is_prob_vec(p; rtol)

Check if `p` is a probability distribution vector.
"""
function is_prob_vec(p::AbstractVector; rtol=1e-2)
    return (minimum(p) >= 0) && isapprox(sum(p), 1; rtol=rtol)
end

function check_prob_vec(p::AbstractVector)
    if !is_prob_vec(p)
        throw(ArgumentError("Invalid probability distribution."))
    end
end

sum_to_one!(x) = normalize!(x, 1)

"""
    rand_prob_vec(rng, N)
    rand_prob_vec(N)

Generate a random probability distribution vector of length `N`.
"""
function rand_prob_vec(rng::AbstractRNG, N)
    p = rand(rng, N)
    sum_to_one!(p)
    return p
end

rand_prob_vec(N) = rand_prob_vec(GLOBAL_RNG, N)
