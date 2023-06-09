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
utils_hmmlearn = pyimport("utils_hmmlearn")
utils_pomegranate = pyimport("utils_pomegranate")

function run_single_python_benchmark(; implem::String, N, D, T, I, number, repeat)
    if implem == "hmmlearn"
        res = utils_hmmlearn.benchmark(; N, D, T, I, number, repeat)
    elseif implem == "pomegranate"
        res = utils_pomegranate.benchmark(; N, D, T, I, number, repeat)
    end
    resv = pyconvert.(Vector, res)
    (logdensity_times, viterbi_times, forward_backward_times, baum_welch_times) = resv
    return (; logdensity_times, viterbi_times, forward_backward_times, baum_welch_times)
end

function run_python_suite(; N_values, D_values, T_values, I, path, number=100, repeat=10)
    @info "Python benchmarks"
    results = Dict()
    algos = ["Logdensity", "Viterbi", "Forward-backward", "Baum-Welch"]
    implems = ["hmmlearn", "pomegranate"]
    for algo in algos
        results[algo] = Dict()
        for implem in implems
            results[algo][implem] = Dict()
        end
    end
    for implem in implems, N in N_values, D in D_values, T in T_values
        println("Benchmarking $implem $((; N, D, T, I))")
        tup = run_single_python_benchmark(; implem, N, D, T, I, number, repeat)
        (; logdensity_times, viterbi_times, forward_backward_times, baum_welch_times) = tup
        results["Logdensity"][implem][(N, D, T, I)] = logdensity_times
        results["Viterbi"][implem][(N, D, T, I)] = viterbi_times
        results["Forward-backward"][implem][(N, D, T, I)] = forward_backward_times
        results["Baum-Welch"][implem][(N, D, T, I)] = baum_welch_times
    end

    if !isnothing(path)
        open(path, "w") do file
            JSON.print(file, results)
        end
    end
    return results
end
