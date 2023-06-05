struct VectorEmissions{D,V<:AbstractVector{D}} <: AbstractEmissions
    distributions::V
end

nb_states(emissions::VectorEmissions) = length(emissions.distributions)
emission_distribution(emissions::VectorEmissions, i::Integer) = emissions.distributions[i]

function reestimate!(
    emissions::VectorEmissions{D}, obs_seqs_concat::Vector, γ_concat::Matrix
) where {D}
    N = nb_states(emissions)
    for i in 1:N
        @views emissions.distributions[i] = fit(D, obs_seqs_concat, γ_concat[i, :])
    end
    return nothing
end
