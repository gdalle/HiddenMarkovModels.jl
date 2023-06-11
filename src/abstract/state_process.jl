"""
    StateProcess

Abstract type for the state part of an HMM.

# Required methods

- `length(sp)`
- `initial_distribution(sp)`
- `transition_matrix(sp)`

# Optional methods

- `fit!(sp, p_count, A_count)`
"""
abstract type StateProcess end

## Interface

"""
    length(sp::StateProcess)

Return the number of states of `sp`.
"""
function Base.length(::SP) where {SP<:StateProcess}
    return error("$SP needs to implement length(sp)")
end

"""
    initial_distribution(sp::StateProcess)

Return the initial state probabilities of `sp`.
"""
function initial_distribution(::SP) where {SP<:StateProcess}
    return error("$SP needs to implement HMMs.initial_distribution(sp)")
end

"""
    transition_matrix(sp::StateProcess)

Return the state transition probabilities of `sp`.
"""
function transition_matrix(::SP) where {SP<:StateProcess}
    return error("$SP needs to implement HMMs.transition_matrix(sp)")
end

"""
    StatsAPI.fit!(op::ObservationProcess, obs_seq, γ)

Update the initial distribution and transition matrix of `sp` based on weighted initialization counts `p_count` and transition counts `A_count`.
"""
function StatsAPI.fit!(::SP, p_count, A_count) where {SP<:StateProcess}
    return error(
        "$SP needs to implement StatsAPI.fit!(sp, p_count, A_count) for Baum-Welch."
    )
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
