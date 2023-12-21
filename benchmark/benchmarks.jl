using BenchmarkTools
using HMMBenchmark
using Random

rng = Random.default_rng()
Random.seed!(rng, 63)

implems = ("HiddenMarkovModels.jl",)
algos = ("rand", "logdensity", "forward", "viterbi", "forward_backward", "baum_welch")
configurations = []
for nb_states in (4, 16, 64)
    push!(
        configurations,
        Configuration(;
            sparse=false, nb_states, obs_dim=1, seq_length=100, nb_seqs=100, bw_iter=1
        ),
    )
end

SUITE = define_suite(rng; implems, configurations, algos)
BenchmarkTools.save(joinpath(@__DIR__, "tune.json"), BenchmarkTools.params(SUITE));
