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
)[1]
results_python = JSON.parsefile(joinpath(@__DIR__, "src", "assets", "results_python.json"))

algos = ["Logdensity", "Viterbi", "Forward-backward", "Baum-Welch"]
julia_implems = ["HMMs.jl", "HMMBase.jl"]
python_implems = ["hmmlearn"]
implems = vcat(julia_implems, python_implems)

linestyles = [:solid, :dash, :dashdot, :dot]
markershapes = [:circle, :square, :diamond]

K = length(implems)

param_tuples = identity.(collect(keys(results_julia[algos[1]][implems[1]])))
N_values = sort(unique(map(t -> t[1], param_tuples)))
D_values = sort(unique(map(t -> t[2], param_tuples)))
T_values = sort(unique(map(t -> t[3], param_tuples)))

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
                leftmargin=D == minimum(D_values) ? 10Plots.mm : -10Plots.mm,
            )
            results_ref = results_julia[algo]["HMMs.jl"]
            for (k, implem) in enumerate(implems)
                if implem in julia_implems
                    results_implem = results_julia[algo][implem]
                    times = [
                        aggregator(results_implem[(N, D, T)].times) / 1e9 for N in N_values
                    ]
                elseif implem in python_implems
                    results_implem = results_python[algo][implem]
                    times = [
                        aggregator(results_implem[string((N, D, T))]) for N in N_values
                    ]
                end
                plot!(
                    plt,
                    N_values,
                    times;
                    label=implem,
                    markershape=markershapes[k],
                    linestyle=linestyles[k],
                )
            end
            push!(plts, plt)
        end
        megaplt = plot(
            plts...;
            size=(1000, 500),
            layout=(1, 3),
            plot_title="$algo (T=$T)",
            legend=:topleft,
            link=:all,
            margin=10Plots.mm,
        )
        if get(ENV, "CI", "false") == "false"
            display(megaplt)
        end
        filename = "benchmark_$(algo)_T=$(T).png"
        savefig(megaplt, joinpath(@__DIR__, "src", "assets", filename))
    end
end
