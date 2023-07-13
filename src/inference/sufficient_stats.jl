
function initialize_states_stats(fbs::Vector{ForwardBackwardStorage{R}}) where {R}
    N = length(first(fbs))
    init_count = Vector{R}(undef, N)
    trans_count = Matrix{R}(undef, N, N)
    return init_count, trans_count
end

function initialize_observations_stats(fbs::Vector{ForwardBackwardStorage{R}}) where {R}
    N = length(first(fbs))
    T_total = sum(duration, fbs)
    state_marginals_concat = Matrix{R}(undef, N, T_total)
    return state_marginals_concat
end

function update_states_stats!(
    init_count, trans_count, fbs::Vector{ForwardBackwardStorage{R}}
) where {R}
    init_count .= zero(R)
    for k in eachindex(fbs)
        @views init_count .+= fbs[k].γ[:, 1]
    end
    trans_count .= zero(R)
    for k in eachindex(fbs)
        sum!(trans_count, fbs[k].ξ; init=false)
    end
    return nothing
end

function update_observations_stats!(
    state_marginals_concat, fbs::Vector{ForwardBackwardStorage{R}}
) where {R}
    T = 1
    for k in eachindex(fbs)
        Tk = duration(fbs[k])
        @views state_marginals_concat[:, T:(T + Tk - 1)] .= fbs[k].γ
        T += Tk
    end
    return nothing
end
