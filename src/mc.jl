struct MarkovChain{U<:AbstractVector,M<:AbstractMatrix} <: AbstractMarkovChain
    init::U
    trans::M

    function MarkovChain(init::U, trans::M) where {U<:AbstractVector,M<:AbstractMatrix}
        mc = new{U,M}(init, trans)
        check_mc(mc)
        return mc
    end
end

const MC = MarkovChain

Base.length(mc::MC) = length(mc.init)
initial_distribution(mc::MC) = mc.init
transition_matrix(mc::MC) = mc.trans

"""
    StatsAPI.fit!(mc::MC, init_count, trans_count)

Update `mc` in-place based on information generated from a state sequence.
"""
function StatsAPI.fit!(mc::MC, init_count, trans_count)
    mc.init .= init_count
    sum_to_one!(mc.init)
    mc.trans .= trans_count
    foreach(sum_to_one!, eachrow(mc.trans))
    return nothing
end
