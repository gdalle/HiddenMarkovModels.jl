## Vector

function likelihoods_vec!(b, op::ObservationProcess, obs)
    for i in 1:length(op)
        b[i] = densityof(distribution(op, i), obs)
    end
    return nothing
end

function loglikelihoods_vec!(logb, op::ObservationProcess, obs)
    for i in 1:length(op)
        logb[i] = logdensityof(distribution(op, i), obs)
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
    T, N = length(obs_seq), length(op)
    B = [densityof(distribution(op, i), obs_seq[t]) for i in 1:N, t in 1:T]
    return B
end

function loglikelihoods(op::ObservationProcess, obs_seq)
    T, N = length(obs_seq), length(op)
    logB = [logdensityof(distribution(op, i), obs_seq[t]) for i in 1:N, t in 1:T]
    return logB
end

## Last argument dispatch

likelihoods(op::ObservationProcess, obs_seq, ::NormalScale) = likelihoods(op, obs_seq)
likelihoods(op::ObservationProcess, obs_seq, ::SemiLogScale) = loglikelihoods(op, obs_seq)
likelihoods(op::ObservationProcess, obs_seq, ::LogScale) = loglikelihoods(op, obs_seq)

function likelihoods!(B, op::ObservationProcess, obs_seq, ::NormalScale)
    return likelihoods!(B, op, obs_seq)
end

function likelihoods!(logB, op::ObservationProcess, obs_seq, ::SemiLogScale)
    return loglikelihoods!(logB, op, obs_seq)
end

function likelihoods!(logB, op::ObservationProcess, obs_seq, ::LogScale)
    return loglikelihoods!(logB, op, obs_seq)
end
