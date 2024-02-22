"""
    rand_prob_vec([rng, ::Type{R},] N)

Generate a random probability distribution of size `N` with normalized uniform entries.
"""
function rand_prob_vec(rng::AbstractRNG, ::Type{R}, N::Integer) where {R}
    p = rand(rng, R, N)
    sum_to_one!(p)
    return p
end

"""
    rand_trans_mat([rng, ::Type{R},] N)

Generate a random transition matrix of size `(N, N)` with normalized uniform entries.
"""
function rand_trans_mat(rng::AbstractRNG, ::Type{R}, N::Integer) where {R}
    A = rand(rng, R, N, N)
    foreach(sum_to_one!, eachrow(A))
    return A
end

rand_prob_vec(rng::AbstractRNG, N::Integer) = rand_prob_vec(rng, Float64, N)
rand_trans_mat(rng::AbstractRNG, N::Integer) = rand_trans_mat(rng, Float64, N)

rand_prob_vec(::Type{R}, N::Integer) where {R} = rand_prob_vec(default_rng(), Float64, N)
rand_trans_mat(::Type{R}, N::Integer) where {R} = rand_trans_mat(default_rng(), Float64, N)

rand_prob_vec(N::Integer) = rand_prob_vec(default_rng(), N)
rand_trans_mat(N::Integer) = rand_trans_mat(default_rng(), N)
