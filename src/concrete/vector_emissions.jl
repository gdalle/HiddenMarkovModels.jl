struct VectorEmissions{D,V<:AbstractVector{D}} <: AbstractEmissions
    distributions::V
end

nb_states(emissions::VectorEmissions) = length(emissions.distributions)
emission_distribution(emissions::VectorEmissions, i::Integer) = emissions.distributions[i]

function reestimate!(
    emissions::VectorEmissions{D},
    forbacks::Vector{<:ForwardBackward},
    obs_seqs::Vector{<:Vector},
) where {D}
    N = nb_states(emissions)
    em = emissions.distributions
    xs = (obs_seqs[k] for k in eachindex(obs_seqs, forbacks))
    @views for i in 1:N
        ws = (forbacks[k].γ[i, :] for k in eachindex(obs_seqs, forbacks))
        em[i] = fit_mle_from_multiple_sequences(D, xs, ws)
    end
    return nothing
end
