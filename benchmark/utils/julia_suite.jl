using Base.Iterators
using BenchmarkTools
using Distributions
using Distributions: PDiagMat
using HMMBase: HMMBase
using HiddenMarkovModels
using JSON
using PythonCall

np = pyimport("numpy")
torch = pyimport("torch")
hmmlearn_hmm = pyimport("hmmlearn.hmm")
pomegranate_hmm = pyimport("pomegranate.hmm")
pomegranate_distributions = pyimport("pomegranate.distributions")

function create_params(; N, D)
    p = ones(N) / N
    A = ones(N, N) / N
    μ = randn(N, D)
    σ = 2 * ones(N, D)
    return (; p, A, μ, σ)
end

function create_numpy_params(; N, D)
    p = np.ones((N,)) / N
    A = np.ones((N, N)) / N
    μ = np.random.randn(N, D)
    σ = 2 * np.ones((N, D))
    return (; p, A, μ, σ)
end

function create_torch_params(; N, D)
    p = torch.ones(N) / N
    A = torch.ones(N, N) / N
    μ = torch.randn(N, D)
    σ = 2 * torch.ones(N, D)
    return (; p, A, μ, σ)
end

function create_dists(; μ, σ)
    N, D = size(μ)
    if D == 1
        dists = [Normal(μ[n, 1], σ[n, 1]) for n in 1:N]
    else
        dists = [DiagNormal(μ[n, :], PDiagMat(σ[n, :] .^ 2)) for n in 1:N]
    end
    return dists
end

function create_obs_seq(; D, T)
    if D == 1
        return randn(T)
    else
        return [randn(D) for t in 1:T]
    end
end

function create_obs_mat(; D, T)
    if D == 1
        return randn(T)
    else
        return randn(T, D)
    end
end

# HiddenMarkovModels.jl

function create_HMMs(; N, D)
    (; p, A, μ, σ) = create_params(; N, D)
    model = HMM(copy(p), copy(A), create_dists(; μ, σ))
    return model
end

function benchmarkables_HMMs(; N, D, T, I)
    create_HMMs(; N=N, D=D)
    obs_seq = create_obs_seq(; D, T)
    obs_seqs = [obs_seq]
    b_logdensity = @benchmarkable logdensityof(model, $obs_seq) setup = (
        model = create_HMMs(; N=$N, D=$D)
    )
    b_viterbi = @benchmarkable viterbi(model, $obs_seq) setup = (
        model = create_HMMs(; N=$N, D=$D)
    )
    b_forward_backward = @benchmarkable forward_backward(model, $obs_seq) setup = (
        model = create_HMMs(; N=$N, D=$D)
    )
    b_baum_welch = @benchmarkable baum_welch(model, $obs_seqs; max_iterations=$I, rtol=-Inf) setup = (
        model = create_HMMs(; N=$N, D=$D)
    )
    return (; b_logdensity, b_viterbi, b_forward_backward, b_baum_welch)
end

# HMMBase.jl

function create_HMMBase(; N, D)
    (; p, A, μ, σ) = create_params(; N, D)
    model = HMMBase.HMM(copy(p), copy(A), create_dists(; μ, σ))
    return model
end

function benchmarkables_HMMBase(; N, D, T, I)
    create_HMMBase(; N=N, D=D)
    obs_mat = create_obs_mat(; D, T)
    b_logdensity = @benchmarkable HMMBase.forward(model, $obs_mat) setup = (
        model = create_HMMBase(; N=$N, D=$D)
    )
    b_viterbi = @benchmarkable HMMBase.viterbi(model, $obs_mat) setup = (
        model = create_HMMBase(; N=$N, D=$D)
    )
    b_forward_backward = @benchmarkable HMMBase.posteriors(model, $obs_mat) setup = (
        model = create_HMMBase(; N=$N, D=$D)
    )
    b_baum_welch = @benchmarkable HMMBase.fit_mle(model, $obs_mat; maxiter=$I, tol=-Inf) setup = (
        model = create_HMMBase(; N=$N, D=$D)
    )
    return (; b_logdensity, b_viterbi, b_forward_backward, b_baum_welch)
end

# hmmlearn

function create_hmmlearn(; N, D, I)
    (; p, A, μ, σ) = create_numpy_params(; N, D)
    model = hmmlearn_hmm.GaussianHMM(;
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

function benchmarkables_hmmlearn(; N, D, T, I)
    create_hmmlearn(; N=N, D=D, I=I)
    obs_mat_py = np.random.randn(T, D)
    b_logdensity = @benchmarkable pycall(model_score, $obs_mat_py) setup = (
        model_score = create_hmmlearn(; N=$N, D=$D, I=$I).score
    )
    b_viterbi = @benchmarkable pycall(model_predict, $obs_mat_py) setup = (
        model_predict = create_hmmlearn(; N=$N, D=$D, I=$I).predict
    )
    b_forward_backward = @benchmarkable pycall(model_predict_proba, $obs_mat_py) setup = (
        model_predict_proba = create_hmmlearn(; N=$N, D=$D, I=$I).predict_proba
    )
    b_baum_welch = @benchmarkable pycall(model_fit, $obs_mat_py) setup = (
        model_fit = create_hmmlearn(; N=$N, D=$D, I=$I).fit
    )
    return (; b_logdensity, b_viterbi, b_forward_backward, b_baum_welch)
end

# pomegranate

function create_pomegranate(; N, D, I)
    (; p, A, μ, σ) = create_torch_params(; N, D)
    distributions = pylist([
        pomegranate_distributions.Normal(;
            means=μ[n], covs=torch.square(σ[n]), covariance_type="diag"
        ) for n in 0:(N - 1)
    ])
    model = pomegranate_hmm.DenseHMM(;
        distributions=distributions, edges=A, starts=p, max_iter=I, tol=1e-10, verbose=false
    )
    return model
end

function benchmarkables_pomegranate(; N, D, T, I)
    create_pomegranate(; N=N, D=D, I=I)
    obs_tens_py = torch.randn(1, T, D)
    b_logdensity = @benchmarkable pycall(model_forward, $obs_tens_py) setup = (
        model_forward = create_pomegranate(; N=$N, D=$D, I=$I).forward
    )
    b_viterbi = @benchmarkable pycall(model_predict, $obs_tens_py) setup = (
        model_predict = create_pomegranate(; N=$N, D=$D, I=$I).predict
    )
    b_forward_backward = @benchmarkable pycall(model_forward_backward, $obs_tens_py) setup = (
        model_forward_backward = create_pomegranate(; N=$N, D=$D, I=$I).forward_backward
    )
    b_baum_welch = @benchmarkable pycall(model_fit, $obs_tens_py) setup = (
        model_fit = create_pomegranate(; N=$N, D=$D, I=$I).fit
    )
    return (; b_logdensity, b_viterbi, b_forward_backward, b_baum_welch)
end

# Suite

function julia_benchmarkables(; implem::String, N, D, T, I)
    if implem == "HMMs.jl"
        return benchmarkables_HMMs(; N, D, T, I)
    elseif implem == "HMMBase.jl"
        return benchmarkables_HMMBase(; N, D, T, I)
    elseif implem == "hmmlearn (jl)"
        return benchmarkables_hmmlearn(; N, D, T, I)
    elseif implem == "pomegranate (jl)"
        return benchmarkables_pomegranate(; N, D, T, I)
    end
end

function define_julia_suite(;
    N_values, D_values, T_values, I, include_python, include_hmmbase
)
    SUITE = BenchmarkGroup()

    algos = ["Logdensity", "Viterbi", "Forward-backward", "Baum-Welch"]
    implems = ["HMMs.jl", "HMMBase.jl", "hmmlearn (jl)", "pomegranate (jl)"]

    for algo in algos
        SUITE[algo] = BenchmarkGroup()
        for implem in implems
            if endswith(implem, ".jl") || include_python
                SUITE[algo][implem] = BenchmarkGroup()
            end
        end
    end

    for implem in implems, N in N_values, D in D_values, T in T_values
        if (
            implem == "HMMs.jl" ||
            (implem == "HMMBase.jl" && include_hmmbase) ||
            (implem in ("hmmlearn (jl)", "pomegranate (jl)") && include_python)
        )
            b_tup = julia_benchmarkables(; implem, N, D, T, I)
            (; b_logdensity, b_viterbi, b_forward_backward, b_baum_welch) = b_tup
            SUITE["Logdensity"][implem][(N, D, T, I)] = b_logdensity
            SUITE["Viterbi"][implem][(N, D, T, I)] = b_viterbi
            SUITE["Forward-backward"][implem][(N, D, T, I)] = b_forward_backward
            SUITE["Baum-Welch"][implem][(N, D, T, I)] = b_baum_welch
        end
    end

    return SUITE
end

function run_julia_suite(;
    N_values, D_values, T_values, I, include_python, include_hmmbase, path, seconds=5
)
    @info "Julia benchmarks"
    SUITE = define_julia_suite(;
        N_values, D_values, T_values, I, include_python, include_hmmbase
    )
    raw_results = run(SUITE; verbose=true, seconds=seconds)
    if !isnothing(path)
        BenchmarkTools.save(path, raw_results)
    end
    return raw_results
end
