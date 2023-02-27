"""
    rand(rng, hmm, T)
    rand(hmm, T)

Simulate `hmm` for `T` time steps and return a named tuple `(state_seq, obs_seq)`.
"""
function Base.rand(rng::AbstractRNG, hmm::AbstractHMM, T::Integer; check_args=false)
    N = nb_states(hmm)
    p = initial_distribution(hmm)
    A = transition_matrix(hmm)
    ems = emission_distributions(hmm)
    state_seq = Vector{Int}(undef, T)
    i = rand(rng, Categorical(p; check_args=check_args))
    state_seq[1] = i
    @views for t in 2:T
        i = rand(rng, Categorical(A[i, :]; check_args=check_args))
        state_seq[t] = i
    end
    obs_seq = [rand(rng, ems[state_seq[t]]) for t in 1:T]
    return (state_seq=state_seq, obs_seq=obs_seq)
end

function Base.rand(hmm::AbstractHMM, T::Integer; check_args=false)
    return rand(GLOBAL_RNG, hmm, T; check_args=check_args)
end
