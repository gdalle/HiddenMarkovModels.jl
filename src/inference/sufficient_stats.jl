
function initialize_states_stats(::Type{R}, hmm::AbstractHMM) where {R}
    init_count = similar(initial_distribution(hmm), R)
    trans_count = similar(transition_matrix(hmm), R)
    return init_count, trans_count
end

function initialize_observations_stats(::Type{R}, hmm::AbstractHMM, obs_seqs) where {R}
    N = length(hmm)
    T_total = sum(length, obs_seqs)
    state_marginals_concat = Matrix{R}(undef, N, T_total)
    return state_marginals_concat
end

function update_states_stats!(
    init_count, trans_count, fbs::Vector{ForwardBackwardStorage{R,M}}
) where {R,M}
    init_count .= zero(R)
    for k in eachindex(fbs)
        init_count .+= fbs[k].γ[1]
    end
    trans_count .= zero(R)
    for k in eachindex(fbs)
        for t in eachindex(fbs[k].ξ)
            mynonzeros(trans_count) .+= mynonzeros(fbs[k].ξ[t])
        end
    end
    return nothing
end

function update_observations_stats!(
    state_marginals_concat, fbs::Vector{ForwardBackwardStorage{R,M}}
) where {R,M}
    T = 0
    for k in eachindex(fbs)
        Tk = duration(fbs[k])
        for t in 1:Tk
            @views state_marginals_concat[:, T + t] .= fbs[k].γ[t]
        end
        T += Tk
    end
    return nothing
end
