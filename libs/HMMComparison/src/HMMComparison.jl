module HMMComparison

using BenchmarkTools
using Distributions
using HMMBase
using HMMBenchmark: benchmarkables_hiddenmarkovmodels, to_tuple
using PythonCall
using Random: AbstractRNG
using SparseArrays

export define_full_suite

function print_python_setup(; path)
    open(path, "w") do file
        redirect_stdout(file) do
            println("Pytorch threads = $(torch.get_num_threads())")
            println("\n# Python packages\n")
        end
        redirect_stderr(file) do
            PythonCall.CondaPkg.status()
        end
    end
end

function define_full_suite(rng::AbstractRNG; implems, configurations, algos)
    SUITE = BenchmarkGroup()
    for implem in implems
        SUITE[implem] = BenchmarkGroup()
        for configuration in configurations
            SUITE[implem][to_tuple(configuration)] = BenchmarkGroup()
            bench_tup = benchmarkables_by_implem(rng; implem, configuration, algos)
            for (algo, bench) in pairs(bench_tup)
                SUITE[implem][to_tuple(configuration)][algo] = bench
            end
        end
    end
    return SUITE
end

function benchmarkables_by_implem(rng::AbstractRNG; implem, configuration, algos)
    if implem == "HiddenMarkovModels.jl"
        return benchmarkables_hiddenmarkovmodels(rng; configuration, algos)
    elseif implem == "HMMBase.jl"
        return benchmarkables_hmmbase(rng; configuration, algos)
    elseif implem == "hmmlearn"
        return benchmarkables_hmmlearn(rng; configuration, algos)
    elseif implem == "pomegranate"
        return benchmarkables_pomegranate(rng; configuration, algos)
    elseif implem == "dynamax"
        return benchmarkables_dynamax(rng; configuration, algos)
    else
        throw(ArgumentError("Unknown implementation"))
    end
end

include("hmmbase.jl")
include("hmmlearn.jl")
include("pomegranate.jl")
include("dynamax.jl")

end # module HMMComparison
