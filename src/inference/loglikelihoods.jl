## Vector

function loglikelihoods_vec!(logb, op::ObservationProcess, obs)
    for i in 1:length(op)
        logb[i] = logdensityof(obs_distribution(op, i), obs)
    end
    check_nan(logb)
    return nothing
end

function loglikelihoods_vec(op::ObservationProcess, obs)
    logb = [logdensityof(obs_distribution(op, i), obs) for i in 1:length(op)]
    check_nan(logb)
    return logb
end

## Matrix

function loglikelihoods!(logB, op::ObservationProcess, obs_seq)
    T, N = length(obs_seq), length(op)
    for t in 1:T, i in 1:N
        logB[i, t] = logdensityof(obs_distribution(op, i), obs_seq[t])
    end
    check_nan(logB)
    return nothing
end

function loglikelihoods(op::ObservationProcess, obs_seq)
    T, N = length(obs_seq), length(op)
    dists = obs_distribution.(Ref(op), 1:N)
    logB = [logdensityof(dists[i], obs_seq[t]) for i in 1:N, t in 1:T]
    check_nan(logB)
    return logB
end
