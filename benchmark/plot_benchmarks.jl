using BenchmarkTools
using Plots
using Statistics

include("utils.jl")

N_values = 2:10
T = 100
baum_welch_iterations = 100

SUITE = define_suite(; N_values, T, baum_welch_iterations)

results = run(SUITE; verbose=true)

# BenchmarkTools.save(joinpath(@__DIR__, "results", "results.json"), results)

for algo in keys(results)
    @info "$algo"
    plt = plot(;
        xlabel="Number of states",
        ylabel="Normalized CPU time",
        title=if algo == "Baum-Welch"
            "$algo ($baum_welch_iterations iter.), T=$T"
        else
            "$algo, T=$T"
        end,
        ylim=(0, 2),
        legend=:topright,
    )
    results_implem_hmmbase = results[algo]["HMMBase.jl"]
    for implem in sort(collect(keys(results[algo])))
        @info " $implem"
        results_implem = results[algo][implem]
        if !isempty(results_implem)
            times_by_N = [
                minimum(results_implem[N].times) / minimum(results_implem_hmmbase[N].times)
                for N in N_values
            ]
            plot!(plt, N_values, times_by_N; linewidth=2, label=implem)
        end
    end
    savefig(plt, joinpath(@__DIR__, "results", "$algo.png"))
end
