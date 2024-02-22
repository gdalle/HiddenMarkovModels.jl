using DataFrames
using Plots
using HMMComparison

data = read_results(joinpath(@__DIR__, "results", "results.csv"))

sort!(data, [:algo, :implem, :nb_states])

implems = [
    "HiddenMarkovModels.jl",  #
    "HMMBase.jl",  #
    "hmmlearn",  #
    "pomegranate",  #
    "dynamax",  #
]
algos = ["viterbi", "forward", "forward_backward", "baum_welch"]

markershapes = [:star5, :circle, :diamond, :hexagon, :pentagon, :utriangle]

for algo in algos
    pl = plot(;
        title=algo,
        size=(800, 400),
        yscale=:log,
        xlabel="nb states",
        ylabel="runtime (s)",
        xticks=unique(data[!, :nb_states]),
        legend=:outerright,
        margin=5Plots.mm,
    )
    for (i, implem) in enumerate(implems)
        subdata = data[(data[!, :algo] .== algo) .& (data[!, :implem] .== implem), :]
        plot!(
            pl,
            subdata[!, :nb_states],
            subdata[!, :time_median] ./ 1e9;
            label=implem,
            markershape=markershapes[i],
            markerstrokecolor=:auto,
            markersize=5,
            linestyle=:auto,
            linewidth=2,
        )
    end
    display(pl)
    savefig(pl, joinpath(@__DIR__, "results", "$(algo).pdf"))
end
