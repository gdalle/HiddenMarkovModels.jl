using BenchmarkTools
using Statistics

include(joinpath(@__DIR__, "benchmarks.jl"))

N_values = 2:20
T = 100
baum_welch_iterations = 10

SUITE = define_suite(; N_values, T, baum_welch_iterations)

raw_results = run(SUITE; verbose=true)
results = median(raw_results)

BenchmarkTools.save(joinpath(@__DIR__, "results.json"), results)
