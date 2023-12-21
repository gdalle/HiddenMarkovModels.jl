using BenchmarkTools
using HMMComparison
using HMMBenchmark
using Random

rng = Random.default_rng()
Random.seed!(rng, 63)

implems = ("HiddenMarkovModels.jl", "HMMBase.jl", "dynamax", "hmmlearn", "pomegranate")
algos = ("logdensity", "baum_welch")
configurations = [
    Configuration(;
        sparse=false, nb_states=4, obs_dim=1, seq_length=100, nb_seqs=100, bw_iter=10
    ),
]

SUITE = define_full_suite(rng; implems, configurations, algos)
# BenchmarkTools.save(joinpath(@__DIR__, "tune.json"), BenchmarkTools.params(SUITE));
results = BenchmarkTools.run(SUITE; verbose=true)
data = parse_results(minimum(results); path=joinpath(@__DIR__, "results.csv"))
