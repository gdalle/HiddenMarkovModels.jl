"""
    logdensityof(hmm, obs_seq)
    logdensityof(hmm, MultiSeq(obs_seqs))

Run the forward algorithm to compute the posterior loglikelihood of sequence `obs_seq` for `hmm`.

This function returns a number.
"""
function DensityInterface.logdensityof(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=no_controls(obs_seq),
)
    _, logL = forward(hmm, obs_seq, control_seq)
    return logL
end

function DensityInterface.logdensityof(
    hmm::AbstractHMM, obs_seqs::MultiSeq, control_seqs::MultiSeq=no_controls(obs_seqs)
)
    return mapreduce(sum, eachindex(sequences(obs_seqs), sequences(control_seqs))) do k
        logdensityof(hmm, obs_seqs[k], control_seqs[k])
    end
end

function logdensityof_with_states(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    state_seq::AbstractVector,
    control_seq::AbstractVector=no_controls(obs_seq),
)
    T = length(eachindex(obs_seq, control_seq))
    R = eltype(hmm, obs_seq[1], control_seq[1])
    logL = zero(R)
    # Initialization
    init = initialization(hmm)
    logL += log(init[state_seq[1]])
    # Transitions
    for t in 1:(T - 1)
        trans = transition_matrix(hmm, control_seq[t])
        logL += log(trans[state_seq[t], state_seq[t + 1]])
    end
    # Observations
    for t in 1:T
        dists = obs_distributions(hmm, control_seq[t])
        logL += logdensityof(dists[state_seq[t]], obs_seq[t])
    end
    return logL
end
