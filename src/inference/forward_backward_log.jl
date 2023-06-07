struct LogForwardBackwardStorage{R}
    logα::Matrix{R}
    logβ::Matrix{R}
    logγ::Matrix{R}
    logξ::Array{R,3}
    _logαA::Matrix{R}  # not temporal
    _logABβ::Matrix{R}  # not temporal
end

Base.length(logfb::LogForwardBackwardStorage) = size(logfb.logα, 1)
duration(logfb::LogForwardBackwardStorage) = size(logfb.logα, 2)

function loglikelihood(logfb::LogForwardBackwardStorage{R}) where {R}
    @views logL = logsumexp(logfb.logα[:, end])
    return logL
end

function loglikelihood(logfbs::Vector{LogForwardBackwardStorage{R}}) where {R}
    logL = zero(R)
    for logfb in logfbs
        logL += loglikelihood(logfb)
    end
    return logL
end

function forward!(logfb::LogForwardBackwardStorage{R}, logp, logA, logB) where {R}
    (; logα, _logαA) = logfb
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

function backward!(logfb::LogForwardBackwardStorage{R}, logA, logB) where {R}
    (; logβ, _logABβ) = logfb
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

function marginals!(logfb::LogForwardBackwardStorage, logA, logB)
    (; logα, logβ, logγ, logξ) = logfb
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

function forward_backward!(logfb::LogForwardBackwardStorage, sp::StateProcess, logB)
    logp = log_initial_distribution(sp)
    logA = log_transition_matrix(sp)
    forward!(logfb, logp, logA, logB)
    backward!(logfb, logA, logB)
    marginals!(logfb, logA, logB)
    return nothing
end

function initialize_forward_backward(sp::StateProcess, logB, ::LogScale)
    N, T = size(logB)
    logp = log_initial_distribution(sp)
    logA = log_transition_matrix(sp)
    R = promote_type(eltype(logp), eltype(logA), eltype(logB))
    logα = Matrix{R}(undef, N, T)
    logβ = Matrix{R}(undef, N, T)
    logγ = Matrix{R}(undef, N, T)
    logξ = Array{R,3}(undef, N, N, T - 1)
    _logαA = Matrix{R}(undef, N, N)
    _logABβ = Matrix{R}(undef, N, N)
    return LogForwardBackwardStorage(logα, logβ, logγ, logξ, _logαA, _logABβ)
end
