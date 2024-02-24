@compile_workload begin
    N, D, T = 3, 2, 10
    init = rand_prob_vec(N)
    trans = rand_trans_mat(N)
    dists = [LightDiagNormal(randn(D), ones(D)) for i in 1:N]
    hmm = HMM(init, trans, dists)
    state_seq, obs_seq = rand(hmm, T)

    logdensityof(hmm, obs_seq)
    joint_logdensityof(hmm, obs_seq, state_seq)
    forward(hmm, obs_seq)
    viterbi(hmm, obs_seq)
    forward_backward(hmm, obs_seq)
    baum_welch(hmm, obs_seq; max_iterations=1)
end
