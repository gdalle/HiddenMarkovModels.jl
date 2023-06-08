using BenchmarkTools
using Distributions: DiagNormal, PDiagMat
using HMMBase: HMMBase
using HiddenMarkovModels
using PythonCall

np = pyimport("numpy")
hmmlearn = pyimport("hmmlearn.hmm")
pomegranate = pyimport("pomegranate")

function to_HMMBase(p, A, μ, σ)
    N = length(p)
    dists = [DiagNormal(μ[n, :], PDiagMat(σ[n, :])) for n in 1:N]
    model = HMMBase.HMM(copy(p), copy(A), dists)
    return model
end

function to_hmmlearn(p, A, μ, σ; max_iterations)
    N = length(p)
    model = hmmlearn.GaussianHMM(;
        n_components=N,
        covariance_type="diag",
        n_iter=max_iterations,
        tol=np.nan,
        implementation="scaling",
        init_params="",
    )
    model.startprob_ = np.array(copy(p))
    model.transmat_ = np.array(copy(A))
    model.means_ = np.array(copy(μ))
    model.covars_ = np.array(copy(σ) .^ 2)
    return model
end

function define_suite(; N_values=2:2:10, D_values=3, T_values=100, max_iterations=10)
    SUITE = BenchmarkGroup()

    for algo in ["Logdensity", "Viterbi", "Forward-backward", "Baum-Welch"]
        SUITE[algo] = BenchmarkGroup()
        for implem in ["HiddenMarkovModels.jl", "HMMBase.jl", "hmmlearn"]
            SUITE[algo][implem] = BenchmarkGroup()
        end
    end

    for N in N_values, D in D_values, T in T_values
        p = rand_prob_vec(N)
        A = rand_trans_mat(N)
        μ = randn(N, D)
        σ = ones(N, D)
        dists = [DiagNormal(μ[n, :], PDiagMat(σ[n, :])) for n in 1:N]
        model = HMM(StandardStateProcess(p, A), StandardObservationProcess(dists))

        (; state_seq, obs_seq) = rand(model, T)
        obs_mat = collect(reduce(hcat, obs_seq)')
        obs_mat_py = np.array(obs_mat)

        # HiddenMarkovModels.jl

        bench = @benchmarkable logdensityof($model, $obs_seq)
        SUITE["Logdensity"]["HiddenMarkovModels.jl"][(N, D, T)] = bench

        bench = @benchmarkable viterbi($model, $obs_seq)
        SUITE["Viterbi"]["HiddenMarkovModels.jl"][(N, D, T)] = bench

        bench = @benchmarkable forward_backward($model, $obs_seq)
        SUITE["Forward-backward"]["HiddenMarkovModels.jl"][(N, D, T)] = bench

        bench = @benchmarkable baum_welch(
            $model, $([obs_seq]); max_iterations=$max_iterations, rtol=NaN
        )
        SUITE["Baum-Welch"]["HiddenMarkovModels.jl"][(N, D, T)] = bench

        # HMMBase.jl

        model_base = to_HMMBase(p, A, μ, σ)

        bench = @benchmarkable HMMBase.forward($model_base, $obs_mat)
        SUITE["Logdensity"]["HMMBase.jl"][(N, D, T)] = bench

        bench = @benchmarkable HMMBase.viterbi($model_base, $obs_mat)
        SUITE["Viterbi"]["HMMBase.jl"][(N, D, T)] = bench

        bench = @benchmarkable HMMBase.posteriors($model_base, $obs_mat)
        SUITE["Forward-backward"]["HMMBase.jl"][(N, D, T)] = bench

        bench = @benchmarkable HMMBase.fit_mle(
            $model_base, $obs_mat; maxiter=$max_iterations, tol=NaN
        )
        SUITE["Baum-Welch"]["HMMBase.jl"][(N, D, T)] = bench

        # hmmlearn

        model_learn = to_hmmlearn(p, A, μ, σ; max_iterations=max_iterations)

        bench = @benchmarkable pycall($(model_learn.score), $obs_mat_py)
        SUITE["Logdensity"]["hmmlearn"][(N, D, T)] = bench

        bench = @benchmarkable pycall($(model_learn.predict), $obs_mat_py)
        SUITE["Viterbi"]["hmmlearn"][(N, D, T)] = bench

        bench = @benchmarkable pycall($(model_learn.predict_proba), $obs_mat_py)
        SUITE["Forward-backward"]["hmmlearn"][(N, D, T)] = bench

        bench = @benchmarkable pycall(model_learn_new.fit, $obs_mat_py) setup = (
            model_learn_new = to_hmmlearn(p, A, μ, σ; max_iterations=max_iterations)
        )
        SUITE["Baum-Welch"]["hmmlearn"][(N, D, T)] = bench
    end

    return SUITE
end

SUITE = define_suite()
