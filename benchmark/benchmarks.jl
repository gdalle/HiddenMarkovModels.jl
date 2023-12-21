using BenchmarkTools
using HMMBenchmark
using Random

rng = Random.default_rng()
Random.seed!(rng, 63)

algos = ("rand", "logdensity", "forward", "viterbi", "forward_backward", "baum_welch")
configurations = [
    # compare state numbers
    Configuration(; nb_states=4, obs_dim=1),
    Configuration(; nb_states=8, obs_dim=1),
    Configuration(; nb_states=16, obs_dim=1),
    Configuration(; nb_states=32, obs_dim=1),
    # compare sparse
    Configuration(; nb_states=64, obs_dim=1, sparse=false),
    Configuration(; nb_states=64, obs_dim=1, sparse=true),
    # compare dists
    Configuration(; nb_states=4, obs_dim=10, custom_dist=true),
    Configuration(; nb_states=4, obs_dim=10, custom_dist=false),
]

SUITE = define_suite(rng; configurations, algos)
BenchmarkTools.save(joinpath(@__DIR__, "tune.json"), BenchmarkTools.params(SUITE));
