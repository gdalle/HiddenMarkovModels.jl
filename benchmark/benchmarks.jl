using HMMBenchmark
using BenchmarkTools

SUITE = define_suite(;
    implems=("HMMs.jl",),
    algos=("logdensity", "viterbi", "forward_backward", "baum_welch"),
    N_vals=4:4:20,
    D_vals=fill(10, 5),
    T_vals=fill(100, 5),
    K_vals=fill(10, 5),
    I_vals=fill(10, 5),
)

BenchmarkTools.save(joinpath(@__DIR__, "tune.json"), BenchmarkTools.params(SUITE));
