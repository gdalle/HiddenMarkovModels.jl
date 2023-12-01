"""
    fit_element_from_sequence!(dists, i, x, w)

Modify the `i`-th element of `dists` by fitting it to an observation sequence `x` with associated weight sequence `w`.

Default behavior:

    fit!(dists[i], x, w)

Specialization for Distributions.jl (in the package extension)

    dists[i] = fit(eltype(dists), x, w)

If this is not possible, please override `fit_element_from_sequence!` directly.
"""
function fit_element_from_sequence!(dists, i, x, w)
    fit!(dists[i], x, w)
    return nothing
end
