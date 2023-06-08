
function initialize_states_stats(fbs::Vector{<:AbstractForwardBackwardStorage{R}}) where {R}
    N = length(first(fbs))
    p_count = Vector{R}(undef, N)
    A_count = Matrix{R}(undef, N, N)
    return p_count, A_count
end

function initialize_observations_stats(
    fbs::Vector{<:AbstractForwardBackwardStorage{R}}
) where {R}
    N = length(first(fbs))
    T_total = sum(duration, fbs)
    γ_concat = Matrix{R}(undef, N, T_total)
    return γ_concat
end

function update_states_stats!(
    p_count,
    A_count,
    fbs::Union{Vector{ForwardBackwardStorage{R}},Vector{SemiLogForwardBackwardStorage{R}}},
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

function update_states_stats!(
    p_count, A_count, logfbs::Vector{LogForwardBackwardStorage{R}}
) where {R}
    N = length(p_count)
    for i in 1:N
        log_p_countᵢ = zero(R)
        for k in eachindex(logfbs)
            @views log_p_countᵢ += logfbs[k].logγ[i, 1]
        end
        p_count[i] = exp(log_p_countᵢ)
    end
    for i in 1:N, j in 1:N
        log_A_countᵢⱼ = zero(R)
        for k in eachindex(logfbs)
            @views log_A_countᵢⱼ += logsumexp(logfbs[k].logξ[i, j, :])
        end
        A_count[i, j] = exp(log_A_countᵢⱼ)
    end
    return nothing
end

function update_observations_stats!(
    γ_concat,
    fbs::Union{Vector{ForwardBackwardStorage{R}},Vector{SemiLogForwardBackwardStorage{R}}},
) where {R}
    T = 1
    for k in eachindex(fbs)
        Tk = duration(fbs[k])
        @views γ_concat[:, T:(T + Tk - 1)] .= fbs[k].γ
        T += Tk
    end
    return nothing
end

function update_observations_stats!(
    γ_concat, logfbs::Vector{LogForwardBackwardStorage{R}}
) where {R}
    T = 1
    for k in eachindex(logfbs)
        Tk = duration(logfbs[k])
        @views γ_concat[:, T:(T + Tk - 1)] .= exp.(logfbs[k].logγ)
        T += Tk
    end
    return nothing
end
