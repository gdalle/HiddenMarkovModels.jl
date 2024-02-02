struct hmmlearnImplem <: Implementation end

function HMMBenchmark.build_model(
    rng::AbstractRNG, implem::hmmlearnImplem; instance::Instance
)
    np = pyimport("numpy")
    hmmlearn_hmm = pyimport("hmmlearn.hmm")

    (; bw_iter, nb_states) = instance
    (; init, trans, means, stds) = build_params(rng; instance)

    hmm = hmmlearn_hmm.GaussianHMM(;
        n_components=nb_states,
        covariance_type="diag",
        n_iter=bw_iter,
        tol=-np.inf,
        implementation="scaling",
        init_params="",
    )

    hmm.startprob_ = np.array(init)
    hmm.transmat_ = np.array(trans)
    hmm.means_ = np.array(transpose(means))
    hmm.covars_ = np.array(transpose(stds .^ 2))
    return hmm
end

function HMMBenchmark.build_benchmarkables(
    rng::AbstractRNG, implem::hmmlearnImplem; instance::Instance, algos::Vector{String}
)
    np = pyimport("numpy")
    (; obs_dim, seq_length, nb_seqs) = instance

    hmm = build_model(rng, implem; instance)
    data = randn(rng, nb_seqs, seq_length, obs_dim)

    obs_mat_concat = reduce(vcat, data[k, :, :] for k in 1:nb_seqs)
    obs_mat_concat_py = np.array(obs_mat_concat)
    obs_mat_len_py = np.full(nb_seqs, seq_length)

    benchs = Dict()

    if "logdensity" in algos
        benchs["logdensity"] = @benchmarkable begin
            $(hmm.score)($obs_mat_concat_py, $obs_mat_len_py)
        end evals = 1 samples = 100
    end

    if "forward" in algos
        benchs["forward"] = @benchmarkable begin
            $(hmm.score)($obs_mat_concat_py, $obs_mat_len_py)
        end evals = 1 samples = 100
    end

    if "viterbi" in algos
        benchs["viterbi"] = @benchmarkable begin
            $(hmm.decode)($obs_mat_concat_py, $obs_mat_len_py)
        end evals = 1 samples = 100
    end

    if "forward_backward" in algos
        benchs["forward_backward"] = @benchmarkable begin
            $(hmm.predict_proba)($obs_mat_concat_py, $obs_mat_len_py)
        end evals = 1 samples = 100
    end

    if "baum_welch" in algos
        benchs["baum_welch"] = @benchmarkable begin
            hmm_guess.fit($obs_mat_concat_py, $obs_mat_len_py)
        end evals = 1 samples = 100 setup = (
            hmm_guess = build_model($rng, $implem; instance=$instance)
        )
    end

    return benchs
end
