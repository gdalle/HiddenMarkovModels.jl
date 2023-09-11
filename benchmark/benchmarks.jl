using Pkg
Pkg.activate(@__DIR__)

using Revise
includet("utils/suite.jl")

SUITE = define_suite(;
    implems=("HMMs.jl",),
    algos=("logdensity", "viterbi", "forward_backward", "baum_welch"),
    N_vals=2:2:10,
    D_vals=(1, 5),
    T_vals=1000,
    K_vals=(1, 10),
)

BenchmarkTools.save(joinpath(@__DIR__, "tune.json"), BenchmarkTools.params(SUITE));
