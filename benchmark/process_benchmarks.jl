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

function plot_benchmarks(; name)
    path = joinpath(DOCS_BENCHMARK_RESULTS_FOLDER, "$(name).csv")
    results = DataFrame(CSV.File(path))

    algos = sort(unique(results[!, :algo]); rev=true)
    implems = sort(unique(results[!, :implem]))

    is_ref = (
        (results[!, :N] .== 1) .&
        (results[!, :D] .== 1) .&
        (results[!, :T] .== 2) .&
        (results[!, :K] .== 1) .&
        (results[!, :I] .== 1)
    )

    ref_results = results[is_ref, :]
    results = results[Bool.(true .- is_ref), :]

    N_vals = sort(unique(results[!, :N]))
    D_vals = sort(unique(results[!, :D]))
    T_vals = sort(unique(results[!, :T]))
    K_vals = sort(unique(results[!, :K]))
    I_vals = sort(unique(results[!, :I]))

    I = only(I_vals)

    for algo in algos, D in D_vals, T in T_vals, K in K_vals
        title_algo = titlecase(replace(algo, "_" => "-"))
        title_params = if algo == "baum_welch"
            "(D=$D,T=$T,K=$K,I=$I)"
        else
            "(D=$D,T=$T,K=$K)"
        end
        plt = plot(;
            title="$title_algo\n$title_params",
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
            local_ref_results = subset(
                ref_results, :algo => x -> x .== algo, :implem => x -> x .== implem
            )
            if size(local_results, 1) == 0
                continue
            end
            sort!(local_results, :N)
            overhead = endswith(implem, ".jl") ? 0 : first(local_ref_results[!, :time])
            plot!(
                plt,
                local_results[!, :N],
                (local_results[!, :time] .- overhead) ./ 1e6;
                label="$implem",
                color=IMPLEM_COLORS[implem],
                markershape=IMPLEM_MARKERSHAPES[implem],
                linestyle=IMPLEM_LINESTYLES[implem],
                linewidth=1.5,
            )
        end

        if get(ENV, "CI", "false") == "false"
            display(plt)
        end

        plot_filename = "$(name)_$(algo)_$(title_params).svg"
        savefig(plt, joinpath(DOCS_BENCHMARK_PLOTS_FOLDER, plot_filename))
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

plot_benchmarks(; name="low_dim")
plot_benchmarks(; name="high_dim")
