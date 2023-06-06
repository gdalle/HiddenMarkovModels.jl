abstract type AbstractTransitions end

## Interface

function nb_states(::Tr) where {Tr<:AbstractTransitions}
    return error("nb_states needs to be implemented for $Tr")
end

function initial_distribution(::Tr) where {Tr<:AbstractTransitions}
    return error("initial_distribution needs to be implemented for $Tr")
end

function transition_matrix(::Tr) where {Tr<:AbstractTransitions}
    return error("transition_matrix needs to be implemented for $Tr")
end

## Checks

function check_transitions(transitions::AbstractTransitions)
    N = nb_states(transitions)
    p = initial_distribution(transitions)
    A = transition_matrix(transitions)
    if !(N > 0)
        throw(ArgumentError("No states"))
    elseif size(p) != (N,)
        throw(DimensionMismatch("Incoherent size for initial distribution"))
    elseif size(A) != (N, N)
        throw(DimensionMismatch("Incoherent size for transition matrix"))
    end
    check_prob_vec(p)
    check_trans_mat(A)
    return nothing
end

## Simulation

function Base.rand(rng::AbstractRNG, transitions::AbstractTransitions, T::Integer)
    p = initial_distribution(transitions)
    A = transition_matrix(transitions)
    state_seq = Vector{Int}(undef, T)
    i = rand(rng, Categorical(p; check_args=false))
    state_seq[1] = i
    for t in 2:T
        @views i = rand(rng, Categorical(A[i, :]; check_args=false))
        state_seq[t] = i
    end
    return state_seq
end

function Base.rand(transitions::AbstractTransitions, T::Integer)
    return rand(GLOBAL_RNG, transitions, T)
end
