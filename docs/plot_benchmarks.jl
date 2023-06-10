using Pkg
Pkg.activate(@__DIR__)

using BenchmarkTools
using JSON
using Plots
using Statistics

BENCHMARKS_FOLDER = joinpath(@__DIR__, "..", "benchmark", "results")
DOCS_FOLDER = joinpath(@__DIR__, "src", "assets")

cp(
    joinpath(BENCHMARKS_FOLDER, "results_julia.json"),
    joinpath(DOCS_FOLDER, "results_julia.json");
    force=true,
)

cp(
    joinpath(BENCHMARKS_FOLDER, "results_python.json"),
    joinpath(DOCS_FOLDER, "results_python.json");
    force=true,
)

results_julia = BenchmarkTools.load(joinpath(DOCS_FOLDER, "results_julia.json"))[1];
results_python = JSON.parsefile(joinpath(DOCS_FOLDER, "results_python.json"));

algos = ["logdensity", "viterbi", "forward_backward", "baum_welch"]
aggregator = minimum

implems_julia = identity.(collect(keys(results_julia)))
implems_python = identity.(collect(keys(results_python)))
implems = sort(vcat(implems_julia, implems_python))
implems = filter(implem -> !contains(implem, "pomegranate"), implems)

param_tuples = identity.(keys(results_julia[implems_julia[1]][algos[1]]))
N_vals = sort(unique(map(t -> t[1], param_tuples)))
D_vals = sort(unique(map(t -> t[2], param_tuples)))
T_vals = sort(unique(map(t -> t[3], param_tuples)))
K_vals = sort(unique(map(t -> t[4], param_tuples)))
I_vals = sort(unique(map(t -> t[5], param_tuples)))

for algo in algos, T in T_vals, K in K_vals, I in I_vals
    plts = []
    for D in D_vals
        plt = plot(;
            xlabel="number of states N",
            title="dimension D=$D",
            ylim=(0, Inf),
            legend=false,
            ylabel=D == minimum(D_vals) ? "$(string(aggregator)) CPU time (s)" : "",
        )
        for implem in implems
            if implem in implems_julia
                local_results = results_julia[implem][algo]
                times = [
                    aggregator(local_results[(N, D, T, K, I)].times) / 1e9 for N in N_vals
                ]
            else
                local_times = results_python[implem][algo]
                times = [aggregator(local_times[string((N, D, T, K, I))]) for N in N_vals]
            end
            plot!(
                plt,
                N_vals,
                times;
                label=implem,
                markershape=:auto,
                linestyle=:auto,
                linewidth=2,
            )
        end
        push!(plts, plt)
    end
    megaplt = plot(
        plts...;
        size=(1000, 500),
        layout=(1, length(plts)),
        plot_title=if algo == "baum_welch"
            "$algo (T=$T, K=$K, I=$I)"
        else
            "$algo (T=$T, K=$K)"
        end,
        legend=:topleft,
        link=:x,
        margin=15Plots.mm,
    )
    if get(ENV, "CI", "false") == "false"
        display(megaplt)
    end
    filename = "benchmark_$(algo)_T=$(T)_K=$(K)_I=$(I).svg"
    savefig(megaplt, joinpath(@__DIR__, "src", "assets", filename))
end
