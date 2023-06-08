using BenchmarkTools
using Statistics

include(joinpath(@__DIR__, "benchmarks.jl"))

SUITE = define_suite()

raw_results = run(SUITE; verbose=true)

BenchmarkTools.save(joinpath(@__DIR__, "results.json"), raw_results)
