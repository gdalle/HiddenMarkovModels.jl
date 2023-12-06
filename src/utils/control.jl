no_controls(T::Integer) = Fill(Nothing, T)
no_controls(obs_seq::AbstractVector) = no_controls(length(obs_seq))
no_controls(obs_seqs::MultiSeq) = MultiSeq(map(no_controls, obs_seqs))
