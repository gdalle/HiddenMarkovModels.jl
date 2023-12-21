using BenchmarkTools
using HMMBenchmark
using Random

rng = Random.default_rng()
Random.seed!(rng, 63)

algos = ("rand", "logdensity", "forward", "viterbi", "forward_backward", "baum_welch")
configurations = []
for nb_states in (4, 16, 64), obs_dim in (1, 100), custom_dist in (true, false)
    push!(
        configurations,
        Configuration(;
            sparse=false,
            custom_dist,
            nb_states,
            obs_dim,
            seq_length=100,
            nb_seqs=100,
            bw_iter=1,
        ),
    )
end

SUITE = define_suite(rng; configurations, algos)
BenchmarkTools.save(joinpath(@__DIR__, "tune.json"), BenchmarkTools.params(SUITE));
