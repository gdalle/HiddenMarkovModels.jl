"""
    logdensityof(hmm, obs_seq)
    logdensityof(hmm, MultiSeq(obs_seqs))

Run the forward algorithm to compute the posterior loglikelihood of sequence `obs_seq` for `hmm`.

This function returns a number.
"""
function DensityInterface.logdensityof(
    hmm::AbstractHMM, obs_seq::Vector, control_seq::AbstractVector=no_controls(obs_seq)
)
    _, logL = forward(hmm, obs_seq, control_seq)
    return logL
end

function DensityInterface.logdensityof(
    hmm::AbstractHMM, obs_seqs::MultiSeq, control_seqs::MultiSeq=no_controls(obs_seqs)
)
    return sum(
        logdensityof(hmm, obs_seqs[k], control_seqs[k]) for
        k in eachindex(sequences(obs_seqs), sequences(control_seqs))
    )
end

"""
    logdensityof(hmm, obs_seq, state_seq)

Compute the joint loglikelihood of sequences `obs_seq` and `state_seq` for `hmm`.

This function returns a number.
"""
function DensityInterface.logdensityof(
    hmm::AbstractHMM,
    obs_seq::Vector,
    state_seq::Vector,
    control_seq::AbstractVector=no_controls(obs_seq),
)
    T = length(obs_seq)
    R = eltype(hmm, obs_seq[1])
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
