## Vector

function likelihoods_vec!(b, op::ObservationProcess, obs)
    for i in 1:length(op)
        b[i] = densityof(distribution(op, i), obs)
    end
    return nothing
end

function loglikelihoods_vec!(logb_vec, op::ObservationProcess, obs)
    for i in 1:length(op)
        logb_vec[i] = logdensityof(distribution(op, i), obs)
    end
    return nothing
end

function likelihoods_vec(op::ObservationProcess, obs)
    b = [densityof(distribution(op, i), obs) for i in 1:length(op)]
    return b
end

function loglikelihoods_vec(op::ObservationProcess, obs)
    logb = [logdensityof(distribution(op, i), obs) for i in 1:length(op)]
    return logb
end

## Matrix

function likelihoods!(B, op::ObservationProcess, obs_seq)
    T, N = length(obs_seq), length(op)
    for t in 1:T, i in 1:N
        B[i, t] = densityof(distribution(op, i), obs_seq[t])
    end
    return nothing
end

function loglikelihoods!(logB, op::ObservationProcess, obs_seq)
    T, N = length(obs_seq), length(op)
    for t in 1:T, i in 1:N
        logB[i, t] = logdensityof(distribution(op, i), obs_seq[t])
    end
    return nothing
end

function likelihoods(op::ObservationProcess, obs_seq)
    em_dists = distributions(op)
    B = densityof.(em_dists, obs_seq')
    return B
end

function loglikelihoods(op::ObservationProcess, obs_seq)
    em_dists = distributions(op)
    logB = logdensityof.(em_dists, obs_seq')
    return logB
end
