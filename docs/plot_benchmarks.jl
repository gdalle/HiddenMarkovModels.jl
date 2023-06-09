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

implems_julia = identity.(collect(keys(results_julia)))
implems_python = identity.(collect(keys(results_python)))
implems = sort(vcat(implems_julia, implems_python))
implems = filter(implem -> implem != "pomegranate", implems)

param_tuples = identity.(keys(results_julia[implems_julia[1]]))
N_values = sort(unique(map(t -> t[1], param_tuples)))
D_values = sort(unique(map(t -> t[2], param_tuples)))
T_values = sort(unique(map(t -> t[3], param_tuples)))
I_values = sort(unique(map(t -> t[4], param_tuples)))

aggregator = minimum

algos = ["logdensity", "viterbi", "forward_backward", "baum_welch"]

for algo in algos, T in T_values, I in I_values
    plts = []
    for D in D_values
        plt = plot(;
            xlabel="number of states N",
            title="dimension D=$D",
            ylim=(0, Inf),
            legend=false,
            ylabel=D == minimum(D_values) ? "$(string(aggregator)) CPU time (s)" : "",
        )
        for implem in implems
            if implem in implems_julia
                times = [
                    aggregator(results_julia[implem][(N, D, T, I)][algo].times) / 1e9 for
                    N in N_values
                ]
            else
                times = [
                    aggregator(results_python[implem][string((N, D, T, I))][algo]) for
                    N in N_values
                ]
            end
            plot!(
                plt,
                N_values,
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
        plot_title=algo == "baum_welch" ? "$algo (T=$T, iter=$I)" : "$algo (T=$T)",
        legend=:topleft,
        link=:x,
        margin=15Plots.mm,
    )
    if get(ENV, "CI", "false") == "false"
        display(megaplt)
    end
    filename = "benchmark_$(algo)_T=$(T)_I=$I.svg"
    savefig(megaplt, joinpath(@__DIR__, "src", "assets", filename))
end
