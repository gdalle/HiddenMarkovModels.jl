"""
    length(model)

Return the number of states of `model`.
"""
Base.length

"""
    initial_distribution(model)

Return the initial state probabilities of `model`.
"""
function initial_distribution end

"""
    transition_matrix(model)

Return the state transition probabilities of `model`.
"""
function transition_matrix end

"""
    rand(rng, model, T)

Simulate `model` for `T` time steps with a specified `rng`.
"""
Base.rand

"""
    rand(model)

Simulate `model` for `T` time steps with the default RNG.
"""
Base.rand(model, T::Integer) = rand(default_rng(), model, T)
