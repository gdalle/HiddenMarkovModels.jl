using JSON
using PythonCall

sys = pyimport("sys")
sys.path.append(@__DIR__)
utils_hmmlearn = pyimport("utils_hmmlearn")
utils_pomegranate = pyimport("utils_pomegranate")

function run_single_python_benchmark(; implem::String, N, D, T, K, I, samples)
    if implem == "hmmlearn"
        res = utils_hmmlearn.benchmark(; N, D, T, K, I, repeat=samples)
    elseif implem == "pomegranate"
        res = utils_pomegranate.benchmark(; N, D, T, K, I, repeat=samples)
    end
    resv = pyconvert.(Vector, res)
    (logdensity, viterbi, forward_backward, baum_welch) = resv
    return (; logdensity, viterbi, forward_backward, baum_welch)
end

function run_python_suite(;
    implems, N_vals, D_vals, T_vals, K_vals, I_vals, samples, path=nothing
)
    @info "Python benchmarks"
    results = Dict()
    for implem in implems
        results[implem] = Dict()
        for N in N_vals, D in D_vals, T in T_vals, K in K_vals, I in I_vals
            println("Benchmarking $implem $((; N, D, T, K, I))")
            times_tup = run_single_python_benchmark(; implem, N, D, T, K, I, samples)
            for (algo, times) in pairs(times_tup)
                if !haskey(results[implem], algo)
                    results[implem][algo] = Dict()
                end
                results[implem][algo][(N, D, T, K, I)] = times
            end
        end
    end
    if !isnothing(path)
        open(path, "w") do file
            JSON.print(file, results)
        end
    end
    return results
end
