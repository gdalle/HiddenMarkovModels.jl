using BenchmarkTools
using HMMComparison
using HMMBenchmark

rng = StableRNG(63)

implems = [
    HiddenMarkovModelsImplem(),  #
    HMMBaseImplem(),  #
    hmmlearnImplem(),  #
    pomegranateImplem(),  #
    dynamaxImplem(),  #
]
algos = ["logdensity", "forward", "viterbi", "forward_backward", "baum_welch"]
instances = [
    Instance(;
        sparse=false, nb_states=5, obs_dim=10, seq_length=100, nb_seqs=50, bw_iter=10
    ),
]

SUITE = define_suite(rng, implems; instances, algos)

results = BenchmarkTools.run(SUITE; verbose=true)
data = parse_results(results)
