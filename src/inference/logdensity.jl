"""
$(SIGNATURES)

Run the forward algorithm to compute the loglikelihood of `obs_seq` for `hmm`, integrating over all possible state sequences.
"""
function DensityInterface.logdensityof(
    hmm::AbstractHMM,
    obs_seq::AbstractVecOrMat,
    control_seq::AbstractVecOrMat=Fill(nothing, duration(obs_seq));
    seq_ends::AbstractVector{Int}=Fill(duration(obs_seq), 1),
)
    _, logL = forward(hmm, obs_seq, control_seq; seq_ends)
    return logL
end

"""
$(SIGNATURES)

Run the forward algorithm to compute the the joint loglikelihood of `obs_seq` and `state_seq` for `hmm`.
"""
function joint_logdensityof(
    hmm::AbstractHMM,
    obs_seq::AbstractVecOrMat,
    state_seq::AbstractVector,
    control_seq::AbstractVecOrMat=Fill(nothing, duration(obs_seq));
    seq_ends::AbstractVector{Int}=Fill(duration(obs_seq), 1),
)
    R = eltype(hmm, obs_seq, control_seq, 1)
    logL = zero(R)
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        # Initialization
        init = initialization(hmm)
        logL += log(init[state_seq[t1]])
        # Transitions
        for t in t1:(t2 - 1)
            trans = transition_matrix(hmm, control_seq, t)
            logL += log(trans[state_seq[t], state_seq[t + 1]])
        end
        # Observations
        for t in t1:t2
            dists = obs_distributions(hmm, control_seq, t)
            logL += logdensityof(dists[state_seq[t]], at_time(obs_seq, t))
        end
    end
    return logL
end
