Base.@kwdef struct Params{T,M<:AbstractMatrix{T}}
    init::Vector{T}
    trans::M
    means::Matrix{T}
    stds::Matrix{T}
end

function build_params(rng::AbstractRNG, instance::Instance)
    (; sparse, nb_states, obs_dim) = instance
    init = ones(nb_states) / nb_states
    if sparse
        trans = spdiagm(
            0 => rand(rng, nb_states) / 2,
            +1 => rand(rng, nb_states - 1) / 2,
            -(nb_states - 1) => rand(rng, 1) / 2,
        )
    else
        trans = rand(rng, nb_states, nb_states)
    end
    for row in eachrow(trans)
        row ./= sum(row)
    end
    means = randn(rng, obs_dim, nb_states)
    stds = ones(obs_dim, nb_states)
    return Params(; init, trans, means, stds)
end
