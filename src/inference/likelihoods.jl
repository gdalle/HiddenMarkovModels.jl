function likelihoods!(B::Matrix, hmm::HMM, obs_seq::Vector)
    T, N = length(obs_seq), nb_states(hmm)
    for i in 1:N
        dist = emission_distribution(hmm, i)
        for t in 1:T
            B[i, t] = densityof(dist, obs_seq[t])
        end
    end
    return nothing
end

function loglikelihoods!(logB::Matrix, hmm::HMM, obs_seq::Vector)
    T, N = length(obs_seq), nb_states(hmm)
    for i in 1:N
        dist = emission_distribution(hmm, i)
        for t in 1:T
            logB[i, t] = logdensityof(dist, obs_seq[t])
        end
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
