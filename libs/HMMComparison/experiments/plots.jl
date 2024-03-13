using DataFrames
using CairoMakie
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

markers = [:star5, :circle, :diamond, :hexagon, :pentagon]
linestyles = [nothing, :dot, :dash, :dashdot, :dashdotdot]

fig = Figure(; size=(900, 700))
axes = []
for (k, algo) in enumerate(algos)
    ax = Axis(
        fig[fld1(k, 2), mod1(k, 2)];
        title=algo,
        xlabel="nb states",
        ylabel="runtime (s)",
        yscale=log10,
        xticks=unique(data[!, :nb_states]),
        yminorticksvisible=true,
        yminorgridvisible=true,
        yminorticks=IntervalsBetween(5),
    )

    for (i, implem) in enumerate(implems)
        subdata = data[(data[!, :algo] .== algo) .& (data[!, :implem] .== implem), :]
        scatterlines!(
            ax,
            subdata[!, :nb_states],
            subdata[!, :time_median] ./ 1e9;
            linewidth=2,
            linestyle=linestyles[i],
            marker=markers[i],
            markersize=15,
            label=implem,
        )
    end
    push!(axes, ax)
end
Legend(fig[3, 1:2], first(axes); orientation=:horizontal)
fig
save(joinpath(@__DIR__, "results", "benchmark.svg"), fig)
