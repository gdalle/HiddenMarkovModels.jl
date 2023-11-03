## Vector

function loglikelihoods_vec!(logb, hmm::AbstractHMM, obs)
    for i in 1:length(hmm)
        logb[i] = logdensityof(obs_distribution(hmm, i), obs)
    end
    check_no_nan(logb)
    check_no_inf(logb)
    return nothing
end

function loglikelihoods_vec(hmm::AbstractHMM, obs)
    testval = logdensityof(obs_distribution(hmm, 1), obs)
    logb = Vector{typeof(testval)}(undef, length(hmm))
    loglikelihoods_vec!(logb, hmm, obs)
    return logb
end

## Matrix

function loglikelihoods!(logB, hmm::AbstractHMM, obs_seq)
    T, N = length(obs_seq), length(hmm)
    for t in 1:T, i in 1:N
        logB[i, t] = logdensityof(obs_distribution(hmm, i), obs_seq[t])
    end
    check_no_nan(logB)
    check_no_inf(logB)
    return nothing
end

function loglikelihoods(hmm::AbstractHMM, obs_seq)
    testval = logdensityof(obs_distribution(hmm, 1), obs_seq[1])
    T, N = length(obs_seq), length(hmm)
    logB = Matrix{typeof(testval)}(undef, N, T)
    loglikelihoods!(logB, hmm, obs_seq)
    return logB
end
