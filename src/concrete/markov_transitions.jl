struct MarkovTransitions{V<:AbstractVector,M<:AbstractMatrix} <: AbstractTransitions
    p::V
    A::M

    function MarkovTransitions(p::V, A::M) where {V<:AbstractVector,M<:AbstractMatrix}
        check_coherent_sizes(p, A)
        check_prob_vec(p)
        check_trans_mat(A)
        return new{V,M}(p, A)
    end
end

nb_states(transitions::MarkovTransitions) = length(transitions.p)
initial_distribution(transitions::MarkovTransitions) = transitions.p
transition_matrix(transitions::MarkovTransitions) = transitions.A

function Base.rand(rng::AbstractRNG, transitions::MarkovTransitions, T::Integer)
    state_seq = Vector{Int}(undef, T)
    i = rand(rng, Categorical(transitions.p; check_args=false))
    state_seq[1] = i
    for t in 2:T
        @views i = rand(rng, Categorical(transitions.A[i, :]; check_args=false))
        state_seq[t] = i
    end
    return state_seq
end

function reestimate!(transitions::MarkovTransitions, p_count::Vector, A_count::Matrix)
    N = nb_states(transitions)
    transitions.p .= p_count ./ sum(p_count)
    transitions.A .= A_count
    for i in 1:N
        @views row_sum = sum(transitions.A[i, :])
        transitions.A[i, :] .*= inv(row_sum)
    end
    check_nan(transitions.p)
    check_nan(transitions.A)
    return nothing
end
