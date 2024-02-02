function build_initialization(rng::AbstractRNG; instance::Instance)
    (; nb_states) = instance
    init = ones(nb_states) / nb_states
    return init
end

function build_transition_matrix(rng::AbstractRNG; instance::Instance)
    (; sparse, nb_states) = instance
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
    return trans
end

function build_means(rng::AbstractRNG; instance::Instance)
    (; obs_dim, nb_states) = instance
    means = randn(rng, obs_dim, nb_states)
    return means
end

function build_stds(rng::AbstractRNG; instance::Instance)
    (; obs_dim, nb_states) = instance
    stds = ones(obs_dim, nb_states)
    return stds
end

function build_params(rng::AbstractRNG; instance::Instance)
    init = build_initialization(rng; instance)
    trans = build_transition_matrix(rng; instance)
    means = build_means(rng; instance)
    stds = build_stds(rng; instance)
    return (; init, trans, means, stds)
end
