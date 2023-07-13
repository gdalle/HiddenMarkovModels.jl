"""
    AbstractMarkovChain

Abstract supertype for a Markov chain amenable to simulation, inference and learning.

# Required interface

- `initial_distribution(mc)`
- `transition_matrix(mc)`
- `fit!(mc, init_count, trans_count)` (optional)

# Applicable methods

- `rand([rng,] mc, T)`
- `logdensityof(mc, obs_seq)`
- `fit(mc, obs_seq)` (if `fit!` is implemented)
"""
abstract type AbstractMarkovChain end

"""
    AbstractMC

Alias for the type `AbstractMarkovChain`.
"""
const AbstractMC = AbstractMarkovChain

@inline DensityInterface.DensityKind(::AbstractMC) = HasDensity()

@required AbstractMC begin
    Base.length(::AbstractMC)
    initial_distribution(::AbstractMC)
    transition_matrix(::AbstractMC)
end

function StatsAPI.fit!(mc::AbstractMC, state_seq::Vector{<:Integer})
    return fit!(mc, [state_seq])
end

function StatsAPI.fit!(mc::AbstractMC, state_seqs::Vector{<:Vector{<:Integer}})
    N = length(mc)
    init_count = zeros(Int, N)
    trans_count = zeros(Int, N, N)
    for state_seq in state_seqs
        init_count[first(state_seq)] += 1
        for t in 1:(length(state_seq) - 1)
            trans_count[state_seq[t], state_seq[t + 1]] += 1
        end
    end
    return fit!(mc, init_count, trans_count)
end

function StatsAPI.fit(mc::AbstractMC, state_seq_or_seqs)
    mc_est = deepcopy(mc)
    fit!(mc_est, state_seq_or_seqs)
    return mc_est
end

"""
    rand(rng, mc, T)

Simulate `mc` for `T` time steps with a specified `rng`.
"""
function Base.rand(rng::AbstractRNG, mc::AbstractMC, T::Integer)
    init = initial_distribution(mc)
    trans = transition_matrix(mc)
    first_state = rand(rng, Categorical(init; check_args=false))
    state_seq = Vector{Int}(undef, T)
    state_seq[1] = first_state
    @views for t in 2:T
        state_seq[t] = rand(rng, Categorical(trans[state_seq[t - 1], :]; check_args=false))
    end
    return state_seq
end

function DensityInterface.logdensityof(mc::AbstractMC, state_seq::Vector{<:Integer})
    init = initial_distribution(mc)
    trans = transition_matrix(mc)
    logL = log(init[first(state_seq)])
    for t in 1:(length(state_seq) - 1)
        logL += log(trans[state_seq[t], state_seq[t + 1]])
    end
    return logL
end
