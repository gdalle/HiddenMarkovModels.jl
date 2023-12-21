function benchmarkables_hmmlearn(rng::AbstractRNG; configuration, algos)
    np = pyimport("numpy")
    (; sparse, nb_states, obs_dim, seq_length, nb_seqs, bw_iter) = configuration

    # Model
    hmm = pyimport("hmmlearn.hmm").GaussianHMM(;
        n_components=nb_states,
        covariance_type="diag",
        n_iter=bw_iter,
        tol=-np.inf,
        implementation="scaling",
        init_params="",
    )
    hmm.startprob_ = np.ones(nb_states) / nb_states
    hmm.transmat_ = np.ones((nb_states, nb_states)) / nb_states
    hmm.means_ =
        np.ones((nb_states, obs_dim)) *
        np.arange(1, nb_states + 1)[0:(nb_states - 1), np.newaxis]
    hmm.covars_ = np.ones((nb_states, obs_dim))

    # Data
    obs_mats_list_py = pylist([hmm.sample(seq_length)[0] for _ in 1:nb_seqs])
    obs_mat_concat_py = np.concatenate(obs_mats_list_py)
    obs_mat_len_py = np.full(nb_seqs, seq_length)

    # Benchmarks
    benchs = Dict()

    if "logdensity" in algos
        benchs["logdensity"] = @benchmarkable begin
            pycall($(hmm.score), $obs_mat_concat_py, $obs_mat_len_py)
        end
    end

    if "forward" in algos
        benchs["forward"] = @benchmarkable begin
            pycall($(hmm.score), $obs_mat_concat_py, $obs_mat_len_py)
        end
    end

    if "viterbi" in algos
        benchs["viterbi"] = @benchmarkable begin
            pycall($(hmm.decode), $obs_mat_concat_py, $obs_mat_len_py)
        end
    end

    if "forward_backward" in algos
        benchs["forward_backward"] = @benchmarkable begin
            pycall($(hmm.predict_proba), $obs_mat_concat_py, $obs_mat_len_py)
        end
    end

    if "baum_welch" in algos
        benchs["baum_welch"] = @benchmarkable begin
            pycall($(hmm.fit), $obs_mat_concat_py, $obs_mat_len_py)
        end
    end

    return benchs
end
