## Vector

function loglikelihoods_vec!(logb, hmm::AbstractHMM, obs)
    for i in 1:length(hmm)
        logb[i] = logdensityof(obs_distribution(hmm, i), obs)
    end
    check_no_nan(logb)
    return nothing
end

function loglikelihoods_vec(hmm::AbstractHMM, obs)
    logb = [logdensityof(obs_distribution(hmm, i), obs) for i in 1:length(hmm)]
    check_no_nan(logb)
    return logb
end

## Matrix

function loglikelihoods!(logB, hmm::AbstractHMM, obs_seq)
    T, N = length(obs_seq), length(hmm)
    for t in 1:T, i in 1:N
        logB[i, t] = logdensityof(obs_distribution(hmm, i), obs_seq[t])
    end
    check_no_nan(logB)
    return nothing
end

function loglikelihoods(hmm::AbstractHMM, obs_seq)
    T, N = length(obs_seq), length(hmm)
    dists = obs_distribution.(Ref(hmm), 1:N)
    logB = [logdensityof(dists[i], obs_seq[t]) for i in 1:N, t in 1:T]
    check_no_nan(logB)
    return logB
end
