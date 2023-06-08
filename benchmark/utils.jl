using Base.Iterators
using BenchmarkTools
using Distributions
using Distributions: PDiagMat
using HMMBase: HMMBase
using HiddenMarkovModels
using JSON
using PythonCall

sys = pyimport("sys")
sys.path.append(@__DIR__)

np = pyimport("numpy")
hmmlearn = pyimport("hmmlearn.hmm")
pomegranate = pyimport("pomegranate")

pyutils = pyimport("pyutils")

function create_params(; N, D)
    p = rand_prob_vec(N)
    A = rand_trans_mat(N)
    μ = randn(N, D)
    σ = ones(N, D)

    p_init = rand_prob_vec(N)
    A_init = rand_trans_mat(N)
    μ_init = randn(N, D)
    σ_init = ones(N, D)
    return (; p, A, μ, σ, p_init, A_init, μ_init, σ_init)
end

# HiddenMarkovModels.jl

function create_HMMs(; p, A, μ, σ)
    N = length(p)
    dists = [DiagNormal(μ[n, :], PDiagMat(σ[n, :])) for n in 1:N]
    model = HMM(copy(p), copy(A), dists)
    return model
end

function benchmarkables_HMMs(; params, T, max_iterations)
    (; p, A, μ, σ, p_init, A_init, μ_init, σ_init) = params
    model = create_HMMs(; p, A, μ, σ)
    (; state_seq, obs_seq) = rand(model, T)
    bench1 = @benchmarkable logdensityof($model, $obs_seq)
    bench2 = @benchmarkable viterbi($model, $obs_seq)
    bench3 = @benchmarkable forward_backward($model, $obs_seq)
    bench4 = @benchmarkable baum_welch(
        model_init, $([obs_seq]); max_iterations=$max_iterations, rtol=-Inf
    ) setup = (model_init = create_HMMs(; p=$p_init, A=$A_init, μ=$μ_init, σ=$σ_init))
    return (bench1, bench2, bench3, bench4)
end

# HMMBase.jl

function create_HMMBase(; p, A, μ, σ)
    N = length(p)
    dists = [DiagNormal(μ[n, :], PDiagMat(σ[n, :])) for n in 1:N]
    model = HMMBase.HMM(copy(p), copy(A), dists)
    return model
end

function benchmarkables_HMMBase(; params, T, max_iterations)
    (; p, A, μ, σ, p_init, A_init, μ_init, σ_init) = params
    model = create_HMMBase(; p, A, μ, σ)
    obs_mat = rand(model, T)
    bench1 = @benchmarkable HMMBase.forward($model, $obs_mat)
    bench2 = @benchmarkable HMMBase.viterbi($model, $obs_mat)
    bench3 = @benchmarkable HMMBase.posteriors($model, $obs_mat)
    bench4 = @benchmarkable HMMBase.fit_mle(
        model_init, $obs_mat; maxiter=$max_iterations, tol=-Inf
    ) setup = (model_init = create_HMMBase(; p=$p_init, A=$A_init, μ=$μ_init, σ=$σ_init))
    return (bench1, bench2, bench3, bench4)
end

# hmmlearn

function create_hmmlearn(; p, A, μ, σ, max_iterations)
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

function benchmarkables_hmmlearn(; params, T, max_iterations)
    (; p, A, μ, σ, p_init, A_init, μ_init, σ_init) = params
    model = create_hmmlearn(; p, A, μ, σ, max_iterations)
    obs_mat_py, state_mat_py = model.sample(T)
    bench1 = @benchmarkable $(model.score)($obs_mat_py)
    bench2 = @benchmarkable $(model.predict)($obs_mat_py)
    bench3 = @benchmarkable $(model.predict_proba)($obs_mat_py)
    bench4 = @benchmarkable model_init.fit($obs_mat_py) setup = (
        model_init = create_hmmlearn(;
            p=$p_init, A=$A_init, μ=$μ_init, σ=$σ_init, max_iterations=$max_iterations
        )
    )
    return (bench1, bench2, bench3, bench4)
end

# Suite

function define_suite(;
    N_values=2:2:10, D_values=3, T_values=100, max_iterations=10, include_python=false
)
    SUITE = BenchmarkGroup()

    for algo in ["Logdensity", "Viterbi", "Forward-backward", "Baum-Welch"]
        SUITE[algo] = BenchmarkGroup()
        for implem in ["HMMs.jl", "HMMBase.jl"]
            SUITE[algo][implem] = BenchmarkGroup()
        end
        if include_python
            for implem in ["hmmlearn"]
                SUITE[algo][implem] = BenchmarkGroup()
            end
        end
    end

    for N in N_values, D in D_values, T in T_values
        params = create_params(; N, D)

        # HiddenMarkovModels.jl

        bench1, bench2, bench3, bench4 = benchmarkables_HMMs(; params, T, max_iterations)
        SUITE["Logdensity"]["HMMs.jl"][(N, D, T)] = bench1
        SUITE["Viterbi"]["HMMs.jl"][(N, D, T)] = bench2
        SUITE["Forward-backward"]["HMMs.jl"][(N, D, T)] = bench3
        SUITE["Baum-Welch"]["HMMs.jl"][(N, D, T)] = bench4

        # HMMBase.jl

        bench1, bench2, bench3, bench4 = benchmarkables_HMMBase(; params, T, max_iterations)
        SUITE["Logdensity"]["HMMBase.jl"][(N, D, T)] = bench1
        SUITE["Viterbi"]["HMMBase.jl"][(N, D, T)] = bench2
        SUITE["Forward-backward"]["HMMBase.jl"][(N, D, T)] = bench3
        SUITE["Baum-Welch"]["HMMBase.jl"][(N, D, T)] = bench4

        # hmmlearn

        if include_python
            bench1, bench2, bench3, bench4 = benchmarkables_hmmlearn(;
                params, T, max_iterations
            )
            SUITE["Logdensity"]["hmmlearn"][(N, D, T)] = bench1
            SUITE["Viterbi"]["hmmlearn"][(N, D, T)] = bench2
            SUITE["Forward-backward"]["hmmlearn"][(N, D, T)] = bench3
            SUITE["Baum-Welch"]["hmmlearn"][(N, D, T)] = bench4
        end
    end

    return SUITE
end

function run_julia_suite(; N_values, D_values, T_values, max_iterations)
    @info "Julia benchmarks"
    SUITE = define_suite(;
        N_values, D_values, T_values, max_iterations, include_python=false
    )
    raw_results = run(SUITE; verbose=true)
    results = median(raw_results)
    BenchmarkTools.save(joinpath(@__DIR__, "results", "results_julia.json"), results)
    return nothing
end

function run_python_suite(; N_values, D_values, T_values, max_iterations)
    @info "Python benchmarks"
    results = Dict()
    for algo in ["Logdensity", "Viterbi", "Forward-backward", "Baum-Welch"]
        results[algo] = Dict()
        for implem in ["hmmlearn"]
            results[algo][implem] = Dict()
        end
    end
    for N in N_values, D in D_values, T in T_values
        println("Benchmarking hmmlearn $((; N, D, T))")
        t = pyutils.benchmark_hmmlearn.benchmark(; N, D, T, max_iterations)
        (logdensity_times, viterbi_times, forward_backward_times, baum_welch_times) =
            pyconvert.(Vector, t)
        results["Logdensity"]["hmmlearn"][(N, D, T)] = median(logdensity_times)
        results["Viterbi"]["hmmlearn"][(N, D, T)] = median(viterbi_times)
        results["Forward-backward"]["hmmlearn"][(N, D, T)] = median(forward_backward_times)
        results["Baum-Welch"]["hmmlearn"][(N, D, T)] = median(baum_welch_times)
    end

    open(joinpath(@__DIR__, "results", "results_python.json"), "w") do f
        JSON.print(f, results)
    end

    return results
end

function run_full_suite(; N_values, D_values, T_values, max_iterations)
    run_julia_suite(; N_values, D_values, T_values, max_iterations)
    run_python_suite(; N_values, D_values, T_values, max_iterations)
    return nothing
end
