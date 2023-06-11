using Pkg
Pkg.activate(@__DIR__)

using BenchmarkTools
using CSV
using DataFrames
using Plots

BENCHMARKS_FOLDER = joinpath(@__DIR__, "..", "benchmark", "results")
DOCS_FOLDER = joinpath(@__DIR__, "src", "assets")

cp(
    joinpath(BENCHMARKS_FOLDER, "results.csv"),
    joinpath(DOCS_FOLDER, "results.csv");
    force=true,
)

cp(joinpath(BENCHMARKS_FOLDER, "setup.txt"), joinpath(DOCS_FOLDER, "setup.txt"); force=true)

results = DataFrame(CSV.File(joinpath(DOCS_FOLDER, "results.csv")))

algos = sort(unique(results[!, :algo]); rev=true)
implems = sort(unique(results[!, :implem]))
filter!(implem -> !contains(implem, "pomegranate"), implems)

N_vals = sort(unique(results[!, :N]))
D_vals = sort(unique(results[!, :D]))
T_vals = sort(unique(results[!, :T]))
K_vals = sort(unique(results[!, :K]))
I_vals = sort(unique(results[!, :I]))

T = maximum(T_vals)
K = maximum(K_vals)
I = maximum(I_vals)

for algo in algos
    plts = []
    for D in D_vals
        plt = plot(;
            xlabel="number of states N",
            title="dimension D=$D",
            ylim=(0, Inf),
            legend=false,
            ylabel=D == minimum(D_vals) ? "min CPU time (s)" : "",
        )
        for implem in implems
            local_results = subset(
                results,
                :algo => x -> x .== algo,
                :implem => x -> x .== implem,
                :D => x -> x .== D,
                :T => x -> x .== T,
                :K => x -> x .== K,
                :I => x -> x .== I,
            )
            sort!(local_results, :N)
            plot!(
                plt,
                local_results[!, :N],
                local_results[!, :time] ./ 1e9;
                label=implem,
                markershape=:auto,
                linestyle=:auto,
                linewidth=2,
                markersize=5,
            )
        end
        push!(plts, plt)
    end

    megaplot_title =
        algo == "baum_welch" ? "$algo (T=$T, K=$K, iter=$I)" : "$algo (T=$T, K=$K)"

    megaplt = plot(
        plts...;
        size=(1000, 500),
        layout=(1, length(plts)),
        plot_title=megaplot_title,
        legend=:topleft,
        link=:x,
        margin=15Plots.mm,
    )
    if get(ENV, "CI", "false") == "false"
        display(megaplt)
    end

    filename = "benchmark_$(algo).svg"
    savefig(megaplt, joinpath(@__DIR__, "src", "assets", filename))
end
