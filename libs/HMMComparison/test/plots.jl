using DataFrames
using Plots

data = read_results(joinpath(@__DIR__, "results.csv"))

algo = "baum_welch"
metric = :time_minimum
data_algo = data[data[!, :algo] .== algo, :]
bar(data_algo[!, :implem], data_algo[!, metric]; title=algo, label=string(metric))
