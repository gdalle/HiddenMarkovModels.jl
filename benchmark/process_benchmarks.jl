using Pkg
Pkg.activate(@__DIR__)

using CSV
using DataFrames
using Plots

BENCHMARK_RESULTS_FOLDER = joinpath(@__DIR__, "results")
DOCS_BENCHMARK_RESULTS_FOLDER = joinpath(
    @__DIR__, "..", "docs", "src", "assets", "benchmark", "results"
)
DOCS_BENCHMARK_PLOTS_FOLDER = joinpath(
    @__DIR__, "..", "docs", "src", "assets", "benchmark", "plots"
)

IMPLEM_COLORS = Dict(
    "HMMs.jl" => :blue, "HMMBase.jl" => :orange, "hmmlearn" => :green, "pomegranate" => :red
)

IMPLEM_MARKERSHAPES = Dict(
    "HMMs.jl" => :circle,
    "HMMBase.jl" => :square,
    "hmmlearn" => :dtriangle,
    "pomegranate" => :utriangle,
)

IMPLEM_LINESTYLES = Dict(
    "HMMs.jl" => :solid,
    "HMMBase.jl" => :dash,
    "hmmlearn" => :dashdot,
    "pomegranate" => :dashdotdot,
)

## Functions

title_from_algo(algo) = titlecase(replace(algo, "_" => "-"))

function plot_benchmarks(; name)
    path = joinpath(DOCS_BENCHMARK_RESULTS_FOLDER, "results_$name.csv")
    results = DataFrame(CSV.File(path))

    algos = sort(unique(results[!, :algo]); rev=true)
    implems = sort(unique(results[!, :implem]))

    N_vals = sort(unique(results[!, :N]))
    D_vals = sort(unique(results[!, :D]))
    T_vals = sort(unique(results[!, :T]))
    K_vals = sort(unique(results[!, :K]))
    I_vals = sort(unique(results[!, :I]))

    D = maximum(D_vals)
    T = maximum(T_vals)
    K = maximum(K_vals)
    I = maximum(I_vals)

    for algo in algos
        title_params =
            algo == "baum_welch" ? "(T=$T, D=$D, K=$K, I=$I)" : "(T=$T, D=$D, K=$K)"
        plt = plot(;
            title=title_from_algo(algo) * "\n$title_params",
            xlim=(1, maximum(N_vals) + 1),
            ylim=(0, Inf),
            legend=:topleft,
            margin=15Plots.mm,
            xlabel="number of states N",
            ylabel="CPU time (ms)",
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
            overhead_results = subset(
                results,
                :algo => x -> x .== algo,
                :implem => x -> x .== implem,
                :N => x -> x .== 1,
                :D => x -> x .== 1,
                :T => x -> x .== 2,
                :K => x -> x .== 1,
                :I => x -> x .== 1,
            )
            overhead = endswith(implem, ".jl") ? 0 : first(overhead_results[!, :time])
            overhead_ms_rounded = round(overhead / 1e6; sigdigits=3)
            plot!(
                plt,
                local_results[!, :N],
                (local_results[!, :time] .- overhead) ./ 1e6;
                label=if endswith(implem, ".jl")
                    "$implem"
                else
                    "$implem [-$(overhead_ms_rounded)ms]"
                end,
                markershape=IMPLEM_MARKERSHAPES[implem],
                color=IMPLEM_COLORS[implem],
                linewidth=1.5,
            )
        end

        if get(ENV, "CI", "false") == "false"
            display(plt)
        end

        filename = "benchmark_$(name)_$(algo).svg"
        savefig(plt, joinpath(DOCS_BENCHMARK_PLOTS_FOLDER, filename))
    end
end

## Main

for file in readdir(BENCHMARK_RESULTS_FOLDER)
    if endswith(file, ".csv") || endswith(file, ".txt")
        cp(
            joinpath(BENCHMARK_RESULTS_FOLDER, file),
            joinpath(DOCS_BENCHMARK_RESULTS_FOLDER, file);
            force=true,
        )
    end
end

plot_benchmarks(; name="single_sequence")
plot_benchmarks(; name="multiple_sequences")
