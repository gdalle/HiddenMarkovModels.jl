using BenchmarkTools
using Plots
using Statistics

cp(
    joinpath(@__DIR__, "..", "benchmark", "results.json"),
    joinpath(@__DIR__, "src", "assets", "results.json");
    force=true,
)

results = median(
    first(BenchmarkTools.load(joinpath(@__DIR__, "src", "assets", "results.json")))
)

algos = ["Logdensity", "Viterbi", "Forward-backward", "Baum-Welch"]
implems = ["HiddenMarkovModels.jl", "HMMBase.jl", "hmmlearn"]
linestyles = [:solid, :dash, :dashdot, :dot]

K = length(implems)

param_tuples = identity.(collect(keys(results[algos[1]][implems[1]])))
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
        results_ref = results[algo]["HiddenMarkovModels.jl"]
        for (k, implem) in enumerate(implems)
            results_implem = results[algo][implem]
            times = [results_implem[(N, D, T)].time for N in N_values]
            bar!(
                plt,
                N_values .+ (((k - 1) - (K ÷ 2)) / 2K),
                times;
                bar_width=0.5 / K,
                label=implem == "HiddenMarkovModels.jl" ? "ours" : implem,
            )
        end
        filename = "benchmark_$(algo)_D=$(D)_T=$(T).png"
        savefig(plt, joinpath(@__DIR__, "src", "assets", filename))
    end
end
