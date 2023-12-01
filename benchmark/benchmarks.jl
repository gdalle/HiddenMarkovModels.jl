using HMMBenchmark
using BenchmarkTools

implems = ("HiddenMarkovModels.jl",)
algos = ("rand", "logdensity", "viterbi", "forward_backward", "baum_welch")
configurations = []
for sparse in (false, true), nb_states in (4, 16, 64)
    push!(
        configurations,
        Configuration(;
            sparse, nb_states, obs_dim=1, seq_length=100, nb_seqs=100, bw_iter=1
        ),
    )
end

SUITE = define_suite(; implems, configurations, algos)
BenchmarkTools.save(joinpath(@__DIR__, "tune.json"), BenchmarkTools.params(SUITE));
