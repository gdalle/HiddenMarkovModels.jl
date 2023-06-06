struct StandardTransitions{V<:AbstractVector,M<:AbstractMatrix} <: AbstractTransitions
    p::V
    A::M

    function StandardTransitions(p::V, A::M) where {V<:AbstractVector,M<:AbstractMatrix}
        check_coherent_sizes(p, A)
        check_prob_vec(p)
        check_trans_mat(A)
        return new{V,M}(p, A)
    end
end

function Base.copy(transitions::StandardTransitions)
    return StandardTransitions(copy(transitions.p), copy(transitions.A))
end

function Base.show(io::IO, transitions::StandardTransitions{V,M}) where {V,M}
    return print(io, "StandardTransitions{$V,$M} with $(nb_states(transitions)) states")
end

nb_states(transitions::StandardTransitions) = length(transitions.p)
initial_distribution(transitions::StandardTransitions) = transitions.p
transition_matrix(transitions::StandardTransitions) = transitions.A

function reestimate!(transitions::StandardTransitions, p_count, A_count)
    transitions.p .= p_count
    sum_to_one!(transitions.p)
    check_nan(transitions.p)

    transitions.A .= A_count
    foreach(sum_to_one!, eachrow(transitions.A))
    check_nan(transitions.A)

    return nothing
end
