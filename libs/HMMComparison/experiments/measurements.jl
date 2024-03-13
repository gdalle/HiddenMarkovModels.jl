@assert Base.Threads.nthreads() == 1

# see https://superfastpython.com/numpy-number-blas-threads/
ENV["MKL_NUM_THREADS"] = 1
ENV["NUMEXPR_NUM_THREADS"] = 1
ENV["OMP_NUM_THREADS"] = 1
ENV["OPENBLAS_NUM_THREADS"] = 1
ENV["VECLIB_MAXIMUM_THREADS"] = 1

# see https://github.com/google/jax/issues/743
ENV["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

using BenchmarkTools
using LinearAlgebra
using PythonCall  # Python process starts now
using StableRNGs
using HMMComparison

# see https://pytorch.org/docs/stable/generated/torch.set_num_threads.html
pyimport("torch").set_num_threads(1)

rng = StableRNG(63)

print_julia_setup(joinpath(@__DIR__, "results", "julia_setup.txt"))
print_python_setup(joinpath(@__DIR__, "results", "python_setup.txt"))

implems = [
    HiddenMarkovModelsImplem(),  #
    HMMBaseImplem(),  #
    hmmlearnImplem(),  #
    pomegranateImplem(),  #
    dynamaxImplem(),  #
]

algos = ["forward", "viterbi", "forward_backward", "baum_welch"]

instances = Instance[]

for nb_states in 2:2:10
    push!(
        instances,
        Instance(;
            custom_dist=false,
            sparse=false,
            nb_states=nb_states,
            obs_dim=1,
            seq_length=100,
            nb_seqs=50,
            bw_iter=5,
        ),
    )
end

SUITE = define_suite(rng, implems; instances, algos)

results = BenchmarkTools.run(SUITE; verbose=true)
data = parse_results(results; path=joinpath(@__DIR__, "results", "results.csv"))
