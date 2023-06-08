using BenchmarkTools
using JSON
using Plots
using Statistics

cp(
    joinpath(@__DIR__, "..", "benchmark", "results", "results_julia.json"),
    joinpath(@__DIR__, "src", "assets", "results_julia.json");
    force=true,
)

cp(
    joinpath(@__DIR__, "..", "benchmark", "results", "results_python.json"),
    joinpath(@__DIR__, "src", "assets", "results_python.json");
    force=true,
)

results_julia = BenchmarkTools.load(
    joinpath(@__DIR__, "src", "assets", "results_julia.json")
)[1]
results_python = JSON.parsefile(joinpath(@__DIR__, "src", "assets", "results_python.json"))

algos = ["Logdensity", "Viterbi", "Forward-backward", "Baum-Welch"]
julia_implems = ["HMMs.jl", "HMMBase.jl"]
python_implems = ["hmmlearn"]
implems = vcat(julia_implems, python_implems)
linestyles = [:solid, :dash, :dashdot, :dot]

K = length(implems)

param_tuples = identity.(collect(keys(results_julia[algos[1]][implems[1]])))
N_values = sort(unique(map(t -> t[1], param_tuples)))
D_values = sort(unique(map(t -> t[2], param_tuples)))
T_values = sort(unique(map(t -> t[3], param_tuples)))

for algo in algos
    for D in D_values, T in T_values
        plt = plot(;
            xlabel="Number of states",
            ylabel="Median CPU time (ns)",
            title="$algo (D=$D, T=$T)",
            ylim=(0, Inf),
            legend=:best,
            margin=10Plots.mm,
        )
        results_ref = results_julia[algo]["HMMs.jl"]
        for (k, implem) in enumerate(implems)
            if implem in julia_implems
                results_implem = results_julia[algo][implem]
                times = [results_implem[(N, D, T)].time for N in N_values]
            elseif implem in python_implems
                results_implem = results_python[algo][implem]
                times = [results_implem[string((N, D, T))] for N in N_values] * 1e9
            end
            bar!(
                plt,
                N_values .+ (((k - 1) - (K ÷ 2)) / 2K),
                times;
                bar_width=0.5 / K,
                label=implem,
            )
        end
        if get(ENV, "CI", "false") == "false"
            display(plt)
        end
        filename = "benchmark_$(algo)_D=$(D)_T=$(T).png"
        savefig(plt, joinpath(@__DIR__, "src", "assets", filename))
    end
end
