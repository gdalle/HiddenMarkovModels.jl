abstract type AbstractEmissions end

@inline DensityInterface.DensityKind(::AbstractEmissions) = HasDensity()

function nb_states(::AbstractEmissions) end
function emission_distribution(::AbstractEmissions, ::Integer) end

function emission_distributions(emissions::AbstractEmissions)
    return [emission_distribution(emissions, i) for i in 1:nb_states(emissions)]
end

function Base.rand(
    rng::AbstractRNG, emissions::AbstractEmissions, state_seq::Vector{<:Integer}
)
    obs_seq = [rand(rng, emission_distribution(emissions, i)) for i in state_seq]
    return obs_seq
end

function Base.rand(emissions::AbstractEmissions, state_seq::Vector{<:Integer})
    return rand(GLOBAL_RNG, emissions, state_seq)
end

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
end
