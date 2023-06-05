abstract type AbstractTransitions end

function nb_states(::AbstractTransitions) end
function initial_distribution(::AbstractTransitions) end
function transition_matrix(::AbstractTransitions) end

function Base.rand(::AbstractRNG, ::AbstractTransitions, T::Integer) end

function Base.rand(transitions::AbstractTransitions, T::Integer)
    return rand(GLOBAL_RNG, transitions, T)
end

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
