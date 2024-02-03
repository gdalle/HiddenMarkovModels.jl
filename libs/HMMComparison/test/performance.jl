using BenchmarkTools
using HMMComparison
using LinearAlgebra
using StableRNGs

BLAS.set_num_threads(1)

rng = StableRNG(63)

implems = [
    HiddenMarkovModelsImplem(),  #
    HMMBaseImplem(),  #
    hmmlearnImplem(),  #
    pomegranateImplem(),  #
    dynamaxImplem(),  #
]

algos = ["forward", "viterbi", "forward_backward", "baum_welch"]

instances = Instance[]

for nb_states in 2:3:24
    push!(
        instances,
        Instance(;
            custom_dist=true,
            sparse=false,
            nb_states=nb_states,
            obs_dim=5,
            seq_length=100,
            nb_seqs=10,
            bw_iter=10,
        ),
    )
end

SUITE = define_suite(rng, implems; instances, algos)

results = BenchmarkTools.run(SUITE; verbose=true)
data = parse_results(results; path=joinpath(@__DIR__, "results.csv"))
