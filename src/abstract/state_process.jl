"""
    StateProcess

Abstract type for the state part of an HMM.

# Required methods

- `Base.length(sp)`
- `initial_distribution(sp)`
- `transition_matrix(sp)`

# Optional methods

- `reestimate!(sp, p_count, A_count)`
"""
abstract type StateProcess end

## Interface

function Base.length(::SP) where {SP<:StateProcess}
    return error("length needs to be implemented for $St")
end

function initial_distribution(::SP) where {SP<:StateProcess}
    return error("initial_distribution needs to be implemented for $St")
end

function transition_matrix(::SP) where {SP<:StateProcess}
    return error("transition_matrix needs to be implemented for $St")
end

## Fallbacks

function initial_distribution!(p, sp::StateProcess)
    p .= initial_distribution(sp)
    return nothing
end

function transition_matrix!(A, sp::StateProcess)
    A .= transition_matrix(sp)
    return nothing
end

## Checks

function check(sp::StateProcess)
    N = length(sp)
    p = initial_distribution(sp)
    A = transition_matrix(sp)
    if !(N > 0)
        throw(ArgumentError("No sp"))
    elseif size(p) != (N,)
        throw(DimensionMismatch("Incoherent size for initial_distribution"))
    elseif size(A) != (N, N)
        throw(DimensionMismatch("Incoherent size for transition_matrix"))
    end
    check_prob_vec(p)
    check_trans_mat(A)
    return nothing
end

## Simulation

function Base.rand(rng::AbstractRNG, sp::StateProcess, T::Integer)
    p = initial_distribution(sp)
    A = transition_matrix(sp)
    state_seq = Vector{Int}(undef, T)
    i = rand(rng, Categorical(p; check_args=false))
    state_seq[1] = i
    for t in 2:T
        @views i = rand(rng, Categorical(A[i, :]; check_args=false))
        state_seq[t] = i
    end
    return state_seq
end

function Base.rand(sp::StateProcess, T::Integer)
    return rand(GLOBAL_RNG, sp, T)
end
