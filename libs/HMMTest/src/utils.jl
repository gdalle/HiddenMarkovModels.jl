function transpose_hmm(hmm::HMM)
    init = initial_distribution(hmm)
    trans = transition_matrix(hmm)
    dists = obs_distributions(hmm)
    trans_transpose = transpose(convert(typeof(trans), transpose(trans)))
    @assert trans_transpose == trans
    return HMM(init, trans_transpose, dists)
end
