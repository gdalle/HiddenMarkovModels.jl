using JSON
using PythonCall

sys = pyimport("sys")
sys.path.append(@__DIR__)
utils_hmmlearn = pyimport("utils_hmmlearn")
utils_pomegranate = pyimport("utils_pomegranate")

function run_single_python_benchmark(; implem::String, N, D, T, I, samples)
    if implem == "hmmlearn"
        res = utils_hmmlearn.benchmark(; N, D, T, I, repeat=samples)
    elseif implem == "pomegranate"
        res = utils_pomegranate.benchmark(; N, D, T, I, repeat=samples)
    end
    resv = pyconvert.(Vector, res)
    (logdensity, viterbi, forward_backward, baum_welch) = resv
    return (; logdensity, viterbi, forward_backward, baum_welch)
end

function run_python_suite(;
    implems, N_values, D_values, T_values, I_values, samples, path=nothing
)
    @info "Python benchmarks"
    results = Dict()
    for implem in implems
        results[implem] = Dict()
        for N in N_values, D in D_values, T in T_values, I in I_values
            println("Benchmarking $implem $((; N, D, T, I))")
            results[implem][(N, D, T, I)] = Dict()
            times_tup = run_single_python_benchmark(; implem, N, D, T, I, samples)
            for (name, times) in pairs(times_tup)
                results[implem][(N, D, T, I)][name] = times
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
