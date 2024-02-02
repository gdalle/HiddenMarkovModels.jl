include("benchmarks.jl")

results = run(SUITE; verbose=true)
data = parse_results(results; path=joinpath(@__DIR__, "results.csv"))
