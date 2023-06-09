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
)[1];
results_python = JSON.parsefile(joinpath(@__DIR__, "src", "assets", "results_python.json"));

algos = identity.(collect(keys(results_julia)))
julia_implems = identity.(collect(keys(results_julia[algos[1]])))
python_implems = identity.(collect(keys(results_python[algos[1]])))
implems = sort(vcat(julia_implems, python_implems))
implems = [implem for implem in implems if !startswith(implem, "pomegranate")]

K = length(implems)

param_tuples = identity.(collect(keys(results_julia[algos[1]][implems[1]])))
N_values = sort(unique(map(t -> t[1], param_tuples)))
D_values = sort(unique(map(t -> t[2], param_tuples)))
T_values = sort(unique(map(t -> t[3], param_tuples)))
I = only(unique(map(t -> t[4], param_tuples)))

aggregator = minimum

for algo in algos
    for T in T_values
        plts = []
        for D in D_values
            plt = plot(;
                xlabel="Number of states",
                title="Dimension D=$D",
                ylim=(0, Inf),
                legend=false,
                ylabel=D == minimum(D_values) ? "$(string(aggregator)) CPU time (s)" : "",
            )
            results_ref = results_julia[algo]["HMMs.jl"]
            for (k, implem) in enumerate(implems)
                if implem in julia_implems
                    results_implem = results_julia[algo][implem]
                    times = [
                        aggregator(results_implem[(N, D, T, I)].times) / 1e9 for
                        N in N_values
                    ]
                elseif implem in python_implems
                    results_implem = results_python[algo][implem]
                    times = [
                        aggregator(results_implem[string((N, D, T, I))]) for N in N_values
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
            plot_title=algo == "Baum-Welch" ? "$algo (T=$T, iter=$I)" : "$algo (T=$T)",
            legend=:topleft,
            link=:all,
            margin=15Plots.mm,
        )
        if get(ENV, "CI", "false") == "false"
            display(megaplt)
        end
        filename = "benchmark_$(algo)_T=$(T).svg"
        savefig(megaplt, joinpath(@__DIR__, "src", "assets", filename))
    end
end
