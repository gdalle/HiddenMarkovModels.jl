struct hmmlearnImplem <: Implementation end
Base.string(::hmmlearnImplem) = "hmmlearn"

function HMMBenchmark.build_model(
    implem::hmmlearnImplem, instance::Instance, params::Params
)
    np = pyimport("numpy")
    hmmlearn_hmm = pyimport("hmmlearn.hmm")

    (; bw_iter, nb_states) = instance
    (; init, trans, means, stds) = params

    hmm = hmmlearn_hmm.GaussianHMM(;
        n_components=nb_states,
        covariance_type="diag",
        n_iter=bw_iter,
        tol=-np.inf,
        implementation="scaling",
        init_params="",
    )

    hmm.startprob_ = Py(init).to_numpy()
    hmm.transmat_ = Py(trans).to_numpy()
    hmm.means_ = Py(transpose(means)).to_numpy()
    hmm.covars_ = Py(transpose(stds .^ 2)).to_numpy()
    return hmm
end

function HMMBenchmark.build_benchmarkables(
    implem::hmmlearnImplem,
    instance::Instance,
    params::Params,
    data::AbstractArray{<:Real,3},
    algos::Vector{String},
)
    (; obs_dim, seq_length, nb_seqs) = instance
    hmm = build_model(implem, instance, params)

    obs_mat_concat = reduce(vcat, data[k, :, :] for k in 1:nb_seqs)
    obs_mat_concat_py = Py(obs_mat_concat).to_numpy()
    obs_mat_len_py = Py(fill(seq_length, nb_seqs)).to_numpy()

    benchs = Dict()

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
            hmm_guess = build_model($implem, $instance, $params)
        )
    end

    return benchs
end
