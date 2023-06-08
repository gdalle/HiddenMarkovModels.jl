using BenchmarkTools
using Plots
using Statistics

cp(
    joinpath(@__DIR__, "..", "benchmark", "results.json"),
    joinpath(@__DIR__, "src", "assets", "benchmark_results.json");
    force=true,
)

results = BenchmarkTools.load(
    joinpath(@__DIR__, "src", "assets", "benchmark_results.json")
)[1]

linestyles = [:solid, :dash, :dashdot, :dot]

for algo in keys(results)
    plt = plot(;
        xlabel="Number of states",
        ylabel="Normalized median CPU time",
        title=algo,
        ylim=(0, 2),
        legend=:topright,
        margin=5Plots.mm,
    )
    results_implem_hmmbase = results[algo]["HMMBase.jl"]
    for (k, implem) in enumerate(sort(collect(keys(results[algo]))))
        results_implem = results[algo][implem]
        if !isempty(results_implem)
            N_values = sort(parse.(Int, collect(keys(results_implem))))
            times_by_N = [
                results_implem[string(N)].time / results_implem_hmmbase[string(N)].time for
                N in N_values
            ]
            plot!(
                plt,
                N_values,
                times_by_N;
                linewidth=2,
                linestyle=linestyles[k],
                markershape=:circle,
                label=implem,
            )
        end
    end
    savefig(plt, joinpath(@__DIR__, "src", "assets", "benchmark_$algo.png"))
end
