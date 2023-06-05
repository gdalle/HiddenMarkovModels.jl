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
        i = rand(rng, Categorical(view(transitions.A, i, :); check_args=false))
        state_seq[t] = i
    end
    return state_seq
end

function reestimate_initial_distribution!(
    transitions::MarkovTransitions, forbacks::Vector{<:ForwardBackward}
)
    p = transitions.p
    p .= zero(eltype(p))
    @views for k in eachindex(forbacks)
        p .+= forbacks[k].γ[:, 1]
    end
    p ./= sum(p)
    check_nan(p)
    return nothing
end

function reestimate_transition_matrix!(
    transitions::MarkovTransitions, forbacks::Vector{<:ForwardBackward}
)
    A = transitions.A
    A .= zero(eltype(A))
    for k in eachindex(forbacks)
        A .+= dropdims(sum(forbacks[k].ξ; dims=3); dims=3)
    end
    A ./= sum(A; dims=2)
    check_nan(A)
    return nothing
end

function reestimate!(transitions::MarkovTransitions, forbacks::Vector{<:ForwardBackward})
    reestimate_initial_distribution!(transitions, forbacks)
    reestimate_transition_matrix!(transitions, forbacks)
    return nothing
end
