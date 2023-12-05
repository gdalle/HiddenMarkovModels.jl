no_controls(T::Integer) = Fill(nothing, T)
no_controls(obs_seq::Vector) = no_controls(length(obs_seq))
no_controls(obs_seqs::MultiSeq) = MultiSeq(map(no_controls ∘ length, obs_seqs))
