@compile_workload begin
    N, D, T = 3, 2, 10
    init = rand_prob_vec(N)
    trans = rand_trans_mat(N)
    dists = [LightDiagNormal(randn(D), ones(D)) for i in 1:N]
    hmm = HMM(init, trans, dists)
    state_seq, obs_seq = rand(hmm, T)
    obs_mat = reduce(hcat, obs_seq)

    for obs_seq_or_mat in (obs_seq, obs_mat)
        logdensityof(hmm, obs_seq_or_mat, state_seq)
        logdensityof(hmm, obs_seq_or_mat)
        forward(hmm, obs_seq_or_mat)
        viterbi(hmm, obs_seq_or_mat)
        forward_backward(hmm, obs_seq_or_mat)
        baum_welch(hmm, obs_seq_or_mat; max_iterations=1)
    end
end
