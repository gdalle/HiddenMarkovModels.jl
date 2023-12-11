module HMMBenchmarkPythonCallExt

using BenchmarkTools
using CondaPkg
using PythonCall
using SimpleUnPack

const torch = pyimport("torch")
const pomegranate = pyimport("pomegranate")
const np = pyimport("numpy")
const hmmlearn = pyimport("hmmlearn")

pyimport("pomegranate.hmm")
pyimport("pomegranate.distributions")
pyimport("hmmlearn.hmm")

function print_python_setup(; path)
    open(path, "w") do file
        redirect_stdout(file) do
            println("Pytorch threads = $(torch.get_num_threads())")
            println("\n# Python packages\n")
        end
        redirect_stderr(file) do
            CondaPkg.status()
        end
    end
end

## hmmlearn

function rand_params_hmmlearn(; N, D)
    p = np.ones((N,)) / N
    A = np.ones((N, N)) / N
    μ = np.random.randn(N, D)
    σ = 2 * np.ones((N, D))
    return (; p, A, μ, σ)
end

function rand_model_hmmlearn(; N, D, I)
    @unpack p, A, μ, σ = rand_params_hmmlearn(; N, D)
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

function benchmarkables_hmmlearn(; algos, N, D, T, K, I)
    rand_model_hmmlearn(; N, D, I)
    obs_mats_list_py = pylist([np.random.randn(T, D) for k in 1:K])
    obs_mat_concat_py = np.concatenate(obs_mats_list_py)
    obs_mat_len_py = np.full(K, T)
    benchs = Dict()
    if "logdensity" in algos
        benchs["logdensity"] = @benchmarkable pycall(
            model_score, $obs_mat_concat_py, $obs_mat_len_py
        ) setup = (model_score = rand_model_hmmlearn(; N=$N, D=$D, I=$I).score)
    end
    if "viterbi" in algos
        benchs["viterbi"] = @benchmarkable pycall(
            model_predict, $obs_mat_concat_py, $obs_mat_len_py
        ) setup = (model_predict = rand_model_hmmlearn(; N=$N, D=$D, I=$I).predict)
    end
    if "forward_backward" in algos
        benchs["forward_backward"] = @benchmarkable pycall(
            model_predict_proba, $obs_mat_concat_py, $obs_mat_len_py
        ) setup = (
            model_predict_proba = rand_model_hmmlearn(; N=$N, D=$D, I=$I).predict_proba
        )
    end
    if "baum_welch" in algos
        benchs["baum_welch"] = @benchmarkable pycall(
            model_fit, $obs_mat_concat_py, $obs_mat_len_py
        ) setup = (model_fit = rand_model_hmmlearn(; N=$N, D=$D, I=$I).fit)
    end
    return benchs
end

## pomegranate

function rand_params_pomegranate(; N, D)
    p = torch.ones(N) / N
    A = torch.ones(N, N) / N
    μ = torch.randn(N, D)
    σ = 2 * torch.ones(N, D)
    return (; p, A, μ, σ)
end

function rand_model_pomegranate(; N, D, I)
    @unpack p, A, μ, σ = rand_params_pomegranate(; N, D)
    distributions = pylist([
        pomegranate.distributions.Normal(;
            means=μ[n], covs=torch.square(σ[n]), covariance_type="diag"
        ) for n in 0:(N - 1)
    ])
    model = pomegranate.hmm.DenseHMM(;
        distributions=distributions, edges=A, starts=p, max_iter=I, tol=1e-10, verbose=false
    )
    return model
end

function benchmarkables_pomegranate(; algos, N, D, T, K, I)
    rand_model_pomegranate(; N, D, I)
    obs_tens_py = torch.randn(K, T, D)
    bench = Dict()
    if "logdensity" in algos
        bench["logdensity"] = @benchmarkable pycall(model_forward, $obs_tens_py) setup = (
            model_forward = rand_model_pomegranate(; N=$N, D=$D, I=$I).forward
        )
    end
    if "forward_backward" in algos
        bench["forward_backward"] = @benchmarkable pycall(
            model_forward_backward, $obs_tens_py
        ) setup = (
            model_forward_backward =
                rand_model_pomegranate(; N=$N, D=$D, I=$I).forward_backward
        )
    end
    if "baum_welch" in algos
        bench["baum_welch"] = @benchmarkable pycall(model_fit, $obs_tens_py) setup = (
            model_fit = rand_model_pomegranate(; N=$N, D=$D, I=$I).fit
        )
    end
    return bench
end

end
