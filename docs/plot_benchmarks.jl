using Pkg

Pkg.activate(joinpath(@__DIR__, "..", "benchmark"))
Pkg.instantiate()

using BenchmarkTools

include(joinpath(@__DIR__, "..", "benchmark", "benchmarks.jl"))

N_values = 2:10
T = 100
baum_welch_iterations = 100

SUITE = define_suite(; N_values, T, baum_welch_iterations)

results = run(SUITE; verbose=get(ENV, "CI", "false") == "false")

BenchmarkTools.save(joinpath(@__DIR__, "assets", "benchmark.json"), results)

Pkg.activate(@__DIR__)

using Plots
using Statistics

linestyles = [:solid, :dash, :dashdot, :dot]

for algo in keys(results)
    @info "$algo"
    plt = plot(;
        xlabel="Number of states",
        ylabel="Normalized median CPU time",
        title=if algo == "Baum-Welch"
            "$algo ($baum_welch_iterations iter.), T=$T"
        else
            "$algo, T=$T"
        end,
        ylim=(0, 2),
        legend=:topright,
    )
    results_implem_hmmbase = results[algo]["HMMBase.jl"]
    for (k, implem) in enumerate(sort(collect(keys(results[algo]))))
        @info " $implem"
        results_implem = results[algo][implem]
        if !isempty(results_implem)
            times_by_N = [
                median(results_implem[N].times) / median(results_implem_hmmbase[N].times)
                for N in N_values
            ]
            plot!(
                plt,
                N_values,
                times_by_N;
                linewidth=2,
                linestyle=linestyles[k],
                label=implem,
            )
        end
    end
    savefig(plt, joinpath(@__DIR__, "assets", "benchmark_$algo.png"))
end
