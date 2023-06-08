## Vector

function loglikelihoods_vec!(logb, op::ObservationProcess, obs)
    for i in 1:length(op)
        logb[i] = logdensityof(distribution(op, i), obs)
    end
    return nothing
end

function loglikelihoods_vec(op::ObservationProcess, obs)
    logb = [logdensityof(distribution(op, i), obs) for i in 1:length(op)]
    return logb
end

## Matrix

function loglikelihoods!(logB, op::ObservationProcess, obs_seq)
    T, N = length(obs_seq), length(op)
    for t in 1:T, i in 1:N
        logB[i, t] = logdensityof(distribution(op, i), obs_seq[t])
    end
    return nothing
end

function loglikelihoods(op::ObservationProcess, obs_seq)
    T, N = length(obs_seq), length(op)
    logB = [logdensityof(distribution(op, i), obs_seq[t]) for i in 1:N, t in 1:T]
    return logB
end
