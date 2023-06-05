function likelihoods!(B::Matrix, hmm::HMM, obs_seq::Vector)
    T, N = length(obs_seq), nb_states(hmm)
    em_dists = emission_distributions(hmm)
    for t in 1:T, i in 1:N
        B[i, t] = densityof(em_dists[i], obs_seq[t])
    end
    return nothing
end

function loglikelihoods!(logB::Matrix, hmm::HMM, obs_seq::Vector)
    T, N = length(obs_seq), nb_states(hmm)
    em_dists = emission_distributions(hmm)
    for t in 1:T, i in 1:N
        logB[i, t] = logdensityof(em_dists[i], obs_seq[t])
    end
    return nothing
end

function likelihoods(hmm::HMM, obs_seq::Vector)
    em_dists = emission_distributions(hmm)
    B = densityof.(em_dists, obs_seq')
    return B
end

function loglikelihoods(hmm::HMM, obs_seq::Vector)
    em_dists = emission_distributions(hmm)
    logB = logdensityof.(em_dists, obs_seq')
    return logB
end
