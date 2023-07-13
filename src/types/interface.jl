"""
    length(model::AbstractModel) 

Return the number of states of `model`.
"""
Base.length

"""
    initial_distribution(model::AbstractModel)

Return the initial state probabilities of `model`.
"""
function initial_distribution end

"""
    transition_matrix(model::AbstractModel) 

Return the state transition probabilities of `model`.
"""
function transition_matrix end

"""
    rand([rng=default_rng(),] model::AbstractModel, T) 

Simulate `model` for `T` time steps with a specified `rng`.
"""
Base.rand(model, T::Integer) = rand(default_rng(), model, T)
