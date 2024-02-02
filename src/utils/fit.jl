"""
$(SIGNATURES)

Modify the `i`-th element of `dists` by fitting it to an observation sequence `x` with associated weight sequence `w`.

Default behavior:

    fit!(dists[i], x, w)

Override for Distributions.jl (in the package extension)

    dists[i] = fit(eltype(dists), x, w)
"""
function fit_in_sequence!(
    dists::AbstractVector, i::Integer, x::AbstractVecOrMat, w::AbstractVector
)
    fit!(dists[i], x, w)
    return nothing
end
