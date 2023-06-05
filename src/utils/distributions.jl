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
    return fit_mle(D, reduce(hcat, x), w)
end

"""
    fit_mle_from_multiple_sequences(::Type{D}, xs, ws)

Fit a distribution of type `D` based on multiple sequences of observations `xs` associated with multiple sequences of weights `ws`.

Must accept arbitrary iterables for `xs` and `ws`.
"""
function fit_mle_from_multiple_sequences(::Type{D}, xs, ws) where {D}
    return fit_mle_from_single_sequence(D, reduce(vcat, xs), reduce(vcat, ws))
end
