function likelihoods!(B::Matrix, hmm::AbstractHMM, θ, obs_seq)
    T, N = length(obs_seq), nb_states(hmm, θ)
    ems = emission_distributions(hmm, θ)
    for t in 1:T, i in 1:1:N
        B[i, t] = densityof(ems[i], obs_seq[t])
    end
    return nothing
end

function loglikelihoods!(logB::Matrix, hmm::AbstractHMM, θ, obs_seq)
    T, N = length(obs_seq), nb_states(hmm, θ)
    ems = emission_distributions(hmm, θ)
    for t in 1:T, i in 1:1:N
        logB[i, t] = logdensityof(ems[i], obs_seq[t])
    end
    return nothing
end

function likelihoods(hmm::AbstractHMM, θ, obs_seq)
    ems = emission_distributions(hmm, θ)
    B = densityof.(ems, obs_seq')
    return B
end

function loglikelihoods(hmm::AbstractHMM, θ, obs_seq)
    ems = emission_distributions(hmm, θ)
    logB = logdensityof.(ems, obs_seq')
    return logB
end
