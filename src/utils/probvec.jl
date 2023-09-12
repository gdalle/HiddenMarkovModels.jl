function is_prob_vec(p::AbstractVector; atol=1e-2)
    return (minimum(p) >= 0) && isapprox(sum(p), 1; atol=atol)
end

sum_to_one!(x) = x ./= sum(x)

function rand_prob_vec(rng::AbstractRNG, N)
    p = rand(rng, N)
    sum_to_one!(p)
    return p
end

rand_prob_vec(N) = rand_prob_vec(default_rng(), N)
