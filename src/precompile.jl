@compile_workload begin
    for dists in (
        [LightCategorical([0.3, 0.7]), LightCategorical([0.8, 0.2])],
        [LightDiagNormal(ones(2), ones(2)), LightDiagNormal(-ones(2), ones(2))],
    )
        init = [0.6, 0.4]
        trans = [0.7 0.3; 0.2 0.8]
        hmm = HMM(init, trans, dists)
        state_seq, obs_seq = rand(hmm, 10)
        state_seq2, obs_seq2 = vcat(state_seq, state_seq), vcat(obs_seq, obs_seq)
        seq_ends = [length(obs_seq), 2length(obs_seq)]

        joint_logdensityof(hmm, obs_seq, state_seq)
        joint_logdensityof(hmm, obs_seq2, state_seq2; seq_ends)
        logdensityof(hmm, obs_seq)
        logdensityof(hmm, obs_seq2; seq_ends)
        viterbi(hmm, obs_seq)
        viterbi(hmm, obs_seq2; seq_ends)
        forward(hmm, obs_seq)
        forward(hmm, obs_seq2; seq_ends)
        baum_welch(hmm, obs_seq)
        baum_welch(hmm, obs_seq2; seq_ends)
    end
end
