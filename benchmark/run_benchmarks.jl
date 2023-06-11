using Pkg
Pkg.activate(@__DIR__)

using Revise
includet("utils/suite.jl")

run_suite(;
    implems=("HMMs.jl", "HMMBase.jl", "hmmlearn", "pomegranate"),
    N_vals=2:2:16,
    D_vals=(1, 5),
    T_vals=1000,
    K_vals=1,
    seconds=5,
    path=joinpath(@__DIR__, "results", "results.csv"),
);

print_setup(; path=joinpath(@__DIR__, "results", "setup.txt"))
