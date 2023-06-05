function initialize_transitions_stats(fbs::MultiForwardBackwardStorage{R}) where {R}
    N = nb_states(first(fbs))
    p_count = Vector{R}(undef, N)
    A_count = Matrix{R}(undef, N, N)
    return p_count, A_count
end

function initialize_emissions_stats(fbs::MultiForwardBackwardStorage{R}) where {R}
    N = nb_states(first(fbs))
    T_total = sum(duration, fbs)
    γ_concat = Matrix{R}(undef, N, T_total)
    return γ_concat
end

function update_transitions_stats!(
    p_count, A_count, fbs::MultiForwardBackwardStorage{R}
) where {R}
    p_count .= zero(R)
    for k in eachindex(fbs)
        @views p_count .+= fbs[k].γ[:, 1]
    end
    A_count .= zero(R)
    for k in eachindex(fbs)
        sum!(A_count, fbs[k].ξ; init=false)
    end
    return nothing
end

function update_emissions_stats!(γ_concat, fbs::MultiForwardBackwardStorage{R}) where {R}
    T = 1
    for k in eachindex(fbs)
        Tk = duration(fbs[k])
        γ_concat[:, T:(T + Tk - 1)] .= fbs[k].γ
        T += Tk
    end
    return nothing
end
