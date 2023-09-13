using Pkg
Pkg.activate(@__DIR__)

using Revise
includet("utils/suite.jl")

print_setup(; path=joinpath(@__DIR__, "results", "setup.txt"))

IMPLEMS_LOW_DIM = ("HMMs.jl", "HMMBase.jl", "hmmlearn")
IMPLEMS_HIGH_DIM = ("HMMs.jl", "hmmlearn", "pomegranate")

ALGOS = ("logdensity", "viterbi", "forward_backward", "baum_welch")

run_suite(;
    implems=IMPLEMS_LOW_DIM,
    algos=ALGOS,
    N_vals=2:3:20,
    D_vals=1,
    T_vals=1000,
    K_vals=1,
    I=10,
    seconds=10,
    samples=20,
    path=joinpath(@__DIR__, "results", "low_dim.csv"),
);

run_suite(;
    implems=IMPLEMS_HIGH_DIM,
    algos=ALGOS,
    N_vals=2:3:20,
    D_vals=10,
    T_vals=200,
    K_vals=50,
    I=10,
    seconds=10,
    samples=20,
    path=joinpath(@__DIR__, "results", "high_dim.csv"),
);
