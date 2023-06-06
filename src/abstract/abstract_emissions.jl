abstract type AbstractEmissions end

## Interface

function nb_states(::Em) where {Em<:AbstractEmissions}
    return error("nb_states needs to be implemented for $Em")
end

function emission_distribution(::Em, ::Integer) where {Em<:AbstractEmissions}
    return error("emission_distribution needs to be implemented for $Em")
end

## Fallbacks

function emission_distributions(emissions::AbstractEmissions)
    return [emission_distribution(emissions, i) for i in 1:nb_states(emissions)]
end

## Checks

function check_emissions(emissions::AbstractEmissions)
    N = nb_states(emissions)
    if !(N > 0)
        throw(ArgumentError("No states in emissions"))
    else
        for i in 1:N
            if DensityKind(emission_distribution(emissions, i)) == NoDensity()
                throw(ArgumentError("Emissions do not satisfy DensityInterface.jl"))
            end
        end
    end
    return nothing
end

## Simulation

function Base.rand(rng::AbstractRNG, emissions::AbstractEmissions, state_seq)
    obs_seq = [rand(rng, emission_distribution(emissions, i)) for i in state_seq]
    return obs_seq
end

function Base.rand(emissions::AbstractEmissions, state_seq)
    return rand(GLOBAL_RNG, emissions, state_seq)
end
