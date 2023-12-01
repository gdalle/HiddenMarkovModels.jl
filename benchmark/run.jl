include("benchmarks.jl")

results = run(SUITE; verbose=true, samples=10)
data = parse_results(minimum(results); path=joinpath(@__DIR__, "results.csv"))
