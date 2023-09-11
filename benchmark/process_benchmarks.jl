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

    T = maximum(T_vals)
    K = maximum(K_vals)
    I = only(I_vals)

    for algo in algos
        plts = []
        for D in D_vals
            plt = plot(;
                xlabel="number of states N",
                title="dimension D=$D\n(T=$T, K=$K)",
                xlim=(0, maximum(N_vals) + 2),
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
                @assert minimum(N_vals) == 1
                if !endswith(implem, ".jl")
                    local_results[!, :time] .-= first(local_results[!, :time])
                end
                plot!(
                    plt,
                    local_results[2:end, :N],
                    local_results[2:end, :time] ./ 1e9;
                    label=implem == "HMMs.jl" ? "$implem (ours)" : implem,
                    markershape=IMPLEM_MARKERSHAPES[implem],
                    color=IMPLEM_COLORS[implem],
                    linestyle=IMPLEM_LINESTYLES[implem],
                    linewidth=1.5,
                )
            end
            push!(plts, plt)
        end

        megaplt = plot(
            plts...;
            size=(length(D_vals) * 500, 500),
            layout=(1, length(plts)),
            plot_title=title_from_algo(algo),
            legend=:topleft,
            link=:all,
            margin=15Plots.mm,
        )
        if get(ENV, "CI", "false") == "false"
            display(megaplt)
        end

        filename = "benchmark_$(name)_$(algo).svg"
        savefig(megaplt, joinpath(DOCS_BENCHMARK_PLOTS_FOLDER, filename))
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
