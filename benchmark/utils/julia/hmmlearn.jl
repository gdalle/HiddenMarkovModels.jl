using BenchmarkTools
using PythonCall

np = pyimport("numpy")
hmmlearn = pyimport("hmmlearn")
pyimport("hmmlearn.hmm")

function rand_params_hmmlearn(; N, D)
    p = np.ones((N,)) / N
    A = np.ones((N, N)) / N
    μ = np.random.randn(N, D)
    σ = 2 * np.ones((N, D))
    return (; p, A, μ, σ)
end

function rand_model_hmmlearn(; N, D, I)
    (; p, A, μ, σ) = rand_params_hmmlearn(; N, D)
    model = hmmlearn.hmm.GaussianHMM(;
        n_components=N,
        covariance_type="diag",
        n_iter=I,
        tol=-np.inf,
        implementation="scaling",
        init_params="",
    )
    model.startprob_ = p
    model.transmat_ = A
    model.means_ = μ
    model.covars_ = np.square(σ)
    return model
end

function benchmarkables_hmmlearn(; N, D, T, K, I)
    rand_model_hmmlearn(; N, D, I)
    obs_mats_list_py = pylist([np.random.randn(T, D) for k in 1:K])
    obs_mat_concat_py = np.concatenate(obs_mats_list_py)
    obs_mat_len_py = np.full(K, T)

    logdensity = @benchmarkable pycall(model_score, $obs_mat_concat_py, $obs_mat_len_py) setup = (
        model_score = rand_model_hmmlearn(; N=$N, D=$D, I=$I).score
    )

    viterbi = @benchmarkable pycall(model_predict, $obs_mat_concat_py, $obs_mat_len_py) setup = (
        model_predict = rand_model_hmmlearn(; N=$N, D=$D, I=$I).predict
    )

    forward_backward = @benchmarkable pycall(
        model_predict_proba, $obs_mat_concat_py, $obs_mat_len_py
    ) setup = (model_predict_proba = rand_model_hmmlearn(; N=$N, D=$D, I=$I).predict_proba)

    baum_welch = @benchmarkable pycall(model_fit, $obs_mat_concat_py, $obs_mat_len_py) setup = (
        model_fit = rand_model_hmmlearn(; N=$N, D=$D, I=$I).fit
    )

    return (; logdensity, viterbi, forward_backward, baum_welch)
end
