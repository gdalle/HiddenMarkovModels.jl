function loglikelihoods!(logB::Matrix, hmm::AbstractHMM, obs_seq)
    T, N = length(obs_seq), nb_states(hmm)
    ems = emission_distributions(hmm)
    for t in 1:T, i in 1:1:N
        logB[i, t] = logdensityof(ems[i], obs_seq[t])
    end
    return nothing
end

function loglikelihoods(hmm::AbstractHMM, obs_seq)
    T, N = length(obs_seq), nb_states(hmm)
    ems = emission_distributions(hmm)
    logB = [logdensityof(ems[i], obs_seq[t]) for i in 1:N, t in 1:T]
    return logB
end
