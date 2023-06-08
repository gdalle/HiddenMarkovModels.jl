
function forward!(fb::LogForwardBackwardStorage{R}, logp, logA, logB) where {R}
    (; logα, _logαA) = fb
    N, T = size(logα)
    @views logα[:, 1] .= logp .+ logB[:, 1]
    @views for t in 1:(T - 1)
        _logαA .= logα[:, t] .+ logA
        for j in 1:N
            logα[j, t + 1] = logsumexp(_logαA[:, j])
        end
        logα[:, t + 1] .+= logB[:, t + 1]
    end
    return nothing
end

function backward!(fb::LogForwardBackwardStorage{R}, logA, logB) where {R}
    (; logβ, _logABβ) = fb
    N, T = size(logβ)
    @views logβ[:, T] .= zero(R)
    @views for t in (T - 1):-1:1
        _logABβ .= logA .+ logB[:, t + 1]' .+ logβ[:, t + 1]'
        for i in 1:N
            logβ[i, t] = logsumexp(_logABβ[i, :])
        end
    end
    return nothing
end

function marginals!(fb::LogForwardBackwardStorage, logA, logB)
    (; logα, logβ, logγ, logξ) = fb
    T = size(logγ, 2)
    @views for t in 1:T
        logγ[:, t] .= logα[:, t] .+ logβ[:, t]
        normalization = logsumexp(logγ[:, t])
        logγ[:, t] .-= normalization
    end
    @views for t in 1:(T - 1)
        logξ[:, :, t] .= logα[:, t] .+ logA .+ logB[:, t + 1]' .+ logβ[:, t + 1]'
        normalization = logsumexp(logξ[:, :, t])
        logξ[:, :, t] .-= normalization
    end
    return nothing
end

function forward_backward!(fb::LogForwardBackwardStorage, sp::StateProcess, logB)
    logp = log_initial_distribution(sp)
    logA = log_transition_matrix(sp)
    forward!(fb, logp, logA, logB)
    backward!(fb, logA, logB)
    marginals!(fb, logA, logB)
    return nothing
end
