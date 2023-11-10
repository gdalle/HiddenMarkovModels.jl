using HMMBenchmark
using BenchmarkTools

SUITE = define_suite(;
    implems=("HMMs.jl",),
    algos=("logdensity", "viterbi", "forward_backward", "baum_welch"),
    N_vals=2:2:20,
    D_vals=10,
    T_vals=100,
    K_vals=10,
    I=10,
)

BenchmarkTools.save(joinpath(@__DIR__, "tune.json"), BenchmarkTools.params(SUITE));
