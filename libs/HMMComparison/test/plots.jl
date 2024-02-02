using DataFrames
using Plots

include("comparison.jl")

data

algo = "forward"
metric = :time_minimum
data_algo = data[data[!, :algo] .== algo, :]
bar(data_algo[!, :implem], data_algo[!, metric]; title=algo, label=string(metric))
