using BenchmarkTools
using HMMBenchmark
using Random
using StableRNGs

rng = StableRNG(63)

algos = ["rand", "logdensity", "forward", "viterbi", "forward_backward", "baum_welch"]
instances = [
    # compare state numbers
    Instance(; nb_states=4, obs_dim=1),
    Instance(; nb_states=8, obs_dim=1),
    Instance(; nb_states=16, obs_dim=1),
    Instance(; nb_states=32, obs_dim=1),
    # compare sparse
    Instance(; nb_states=64, obs_dim=1, sparse=false),
    Instance(; nb_states=64, obs_dim=1, sparse=true),
    # compare dists
    Instance(; nb_states=4, obs_dim=10, custom_dist=true),
    Instance(; nb_states=4, obs_dim=10, custom_dist=false),
]

SUITE = define_suite(rng; instances, algos)
