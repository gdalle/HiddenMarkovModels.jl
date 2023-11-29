@compile_workload begin
    N, D, T = 3, 2, 100
    p = rand_prob_vec(N)
    A = rand_trans_mat(N)
    dists = [LightDiagNormal(randn(D), ones(D)) for i in 1:N]
    hmm = HMM(p, A, dists)
    obs_seq = rand(hmm, T).obs_seq

    logdensityof(hmm, obs_seq)
    forward(hmm, obs_seq)
    viterbi(hmm, obs_seq)
    forward_backward(hmm, obs_seq)
    baum_welch(hmm, obs_seq; max_iterations=2, atol=-Inf)
end
