## Vector

function likelihoods_vec!(b, op::ObservationProcess, obs)
    for i in 1:length(op)
        b[i] = densityof(distribution(op, i), obs)
    end
    check_nan(b)
    return nothing
end

function loglikelihoods_vec!(logb, op::ObservationProcess, obs)
    for i in 1:length(op)
        logb[i] = logdensityof(distribution(op, i), obs)
    end
    check_nan(logb)
    return nothing
end

function likelihoods_vec(op::ObservationProcess, obs)
    b = [densityof(distribution(op, i), obs) for i in 1:length(op)]
    check_nan(b)
    return b
end

function loglikelihoods_vec(op::ObservationProcess, obs)
    logb = [logdensityof(distribution(op, i), obs) for i in 1:length(op)]
    check_nan(logb)
    return logb
end

## Matrix

function likelihoods!(B, op::ObservationProcess, obs_seq)
    T, N = length(obs_seq), length(op)
    for t in 1:T, i in 1:N
        B[i, t] = densityof(distribution(op, i), obs_seq[t])
    end
    check_nan(B)
    return nothing
end

function loglikelihoods!(logB, op::ObservationProcess, obs_seq)
    T, N = length(obs_seq), length(op)
    for t in 1:T, i in 1:N
        logB[i, t] = logdensityof(distribution(op, i), obs_seq[t])
    end
    check_nan(logB)
    return nothing
end

function likelihoods(op::ObservationProcess, obs_seq)
    T, N = length(obs_seq), length(op)
    B = [densityof(distribution(op, i), obs_seq[t]) for i in 1:N, t in 1:T]
    check_nan(B)
    return B
end

function loglikelihoods(op::ObservationProcess, obs_seq)
    T, N = length(obs_seq), length(op)
    logB = [logdensityof(distribution(op, i), obs_seq[t]) for i in 1:N, t in 1:T]
    check_nan(logB)
    return logB
end
