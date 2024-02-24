function valid_prob_vec(p::AbstractVector{T}) where {T}
    return minimum(p) >= zero(T) && sum(p) ≈ one(T)
end

function valid_trans_mat(A::AbstractMatrix)
    return size(A, 1) == size(A, 2) && all(valid_prob_vec, eachrow(A))
end

function valid_dists(d::AbstractVector)
    for i in eachindex(d)
        if DensityKind(d[i]) == NoDensity()
            return false
        end
    end
    return true
end

"""
    valid_hmm(hmm)

Perform some checks to rule out obvious inconsistencies with an `AbstractHMM` object.
"""
function valid_hmm(hmm::AbstractHMM, control=nothing)
    init = initialization(hmm)
    trans = transition_matrix(hmm, control)
    dists = obs_distributions(hmm, control)
    if !(length(init) == length(dists) == size(trans, 1) == size(trans, 2))
        return false
    elseif !valid_prob_vec(init)
        return false
    elseif !valid_trans_mat(trans)
        return false
    elseif !valid_dists(dists)
        return false
    end
    return true
end
