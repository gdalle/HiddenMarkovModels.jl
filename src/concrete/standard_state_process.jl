"""
    StandardStateProcess <: StateProcess

# Fields

- `p::AbstractVector`: initial distribution
- `A::AbstractMatrix`: transition matrix
"""
struct StandardStateProcess{V<:AbstractVector,M<:AbstractMatrix} <: StateProcess
    p::V
    A::M

    function StandardStateProcess(p::V, A::M) where {V<:AbstractVector,M<:AbstractMatrix}
        check_coherent_sizes(p, A)
        check_prob_vec(p)
        check_trans_mat(A)
        return new{V,M}(p, A)
    end
end

function Base.copy(sp::StandardStateProcess)
    return StandardStateProcess(copy(sp.p), copy(sp.A))
end

function Base.show(io::IO, sp::StandardStateProcess{V,M}) where {V,M}
    return print(io, "StandardStateProcess{$V,$M} with $(length(sp)) states")
end

Base.length(sp::StandardStateProcess) = length(sp.p)
initial_distribution(sp::StandardStateProcess) = sp.p
transition_matrix(sp::StandardStateProcess) = sp.A

function reestimate!(sp::StandardStateProcess, p_count, A_count)
    sp.p .= p_count
    sum_to_one!(sp.p)
    check_nan(sp.p)
    sp.A .= A_count
    foreach(sum_to_one!, eachrow(sp.A))
    check_nan(sp.A)
    return nothing
end
