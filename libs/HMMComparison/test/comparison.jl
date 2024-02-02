using BenchmarkTools
using HMMComparison
using HMMBenchmark
using Random

rng = Random.default_rng()
Random.seed!(rng, 63)

implems = [
    HiddenMarkovModelsImplem(),  #
    HMMBaseImplem(),  #
    hmmlearnImplem(),  #
    pomegranateImplem(),  #
]
algos = ["logdensity", "forward", "viterbi", "forward_backward", "baum_welch"]
instances = [
    Instance(; sparse=false, nb_states=4, obs_dim=2, seq_length=100, nb_seqs=5, bw_iter=10)
]

SUITE = define_suite(rng, implems; instances, algos)

results = BenchmarkTools.run(SUITE; verbose=true)
data = parse_results(minimum(results))
