"""
    is_prob_vec(p; atol)

Check if `p` is a probability distribution vector.
"""
function is_prob_vec(p::AbstractVector; atol=1e-5)
    return all(>=(0), p) && isapprox(sum(p), 1; atol=atol)
end

"""
    rand_prob_vec(rng, N)
    rand_prob_vec(N)

Generate a random probability distribution vector of length `N`.
"""
function rand_prob_vec(rng::AbstractRNG, N)
    p = rand(rng, N)
    p ./= sum(p)
    return p
end

rand_prob_vec(N) = rand_prob_vec(GLOBAL_RNG, N)

"""
    is_square(A)

Check if `A` is a square matrix.
"""
function is_square(A::AbstractMatrix)
    return size(A, 1) == size(A, 2)
end

"""
    is_trans_mat(A; atol)

Check if `A` is a transition matrix.
"""
function is_trans_mat(A::AbstractMatrix; atol=1e-5)
    if !is_square(A)
        return false
    else
        @views for i in axes(A, 1)
            if !is_prob_vec(A[i, :]; atol=atol)
                return false
            end
        end
        return true
    end
end

"""
    rand_trans_mat(rng, N)
    rand_trans_mat(N)

Generate a random transition matrix of size `N × N`.
"""
function rand_trans_mat(rng::AbstractRNG, N)
    A = rand(rng, N, N)
    A ./= sum(A; dims=2)
    return A
end

rand_trans_mat(N) = rand_trans_mat(GLOBAL_RNG, N)

function check_nan(x::AbstractArray)
    if any(isnan, x)
        throw(OverflowError("Some values are NaNs"))
    end
end

"""
    fit_mle_from_single_sequence(::Type{D}, x, w)

Fit a distribution of type `D` based on a single sequence of observations `x` associated with a single sequence of weights `w`.

Defaults to `Distributions.fit_mle`, with a special case for vectors of vectors (because `Distributions.fit_mle` accepts matrices instead).
Users are free to override this default for concrete distributions.
"""
function fit_mle_from_single_sequence(
    ::Type{D}, x::AbstractVector, w::AbstractVector
) where {D}
    return fit_mle(D, x, w)
end

function fit_mle_from_single_sequence(
    ::Type{D}, x::AbstractVector{<:AbstractVector}, w::AbstractVector
) where {D}
    return fit_mle(D, hcat(x...), w)
end

"""
    fit_mle_from_multiple_sequences(::Type{D}, xs, ws)

Fit a distribution of type `D` based on multiple sequences of observations `xs` associated with multiple sequences of weights `ws`.

Must accept arbitrary iterables for `xs` and `ws`.
"""
function fit_mle_from_multiple_sequences(::Type{D}, xs, ws) where {D}
    return fit_mle_from_single_sequence(D, vcat(xs...), vcat(ws...))
end
