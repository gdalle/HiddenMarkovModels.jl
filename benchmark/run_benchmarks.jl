using Pkg
Pkg.activate(@__DIR__)

using Revise
includet("utils/suite.jl")

print_setup(; path=joinpath(@__DIR__, "results", "setup.txt"))

IMPLEMS_SINGLE_SEQUENCE = ("HMMs.jl", "HMMBase.jl", "hmmlearn")
IMPLEMS_MULTIPLE_SEQUENCES = ("HMMs.jl", "hmmlearn", "pomegranate")

ALGOS_SINGLE_SEQUENCE = ("logdensity", "viterbi", "forward_backward", "baum_welch")
ALGOS_MULTIPLE_SEQUENCES = ("logdensity", "baum_welch")

run_suite(;
    implems=IMPLEMS_SINGLE_SEQUENCE,
    algos=ALGOS_SINGLE_SEQUENCE,
    N_vals=2:2:20,
    D_vals=(1, 10),
    T_vals=1000,
    K_vals=1,
    seconds=10,
    samples=100,
    path=joinpath(@__DIR__, "results", "results_single_sequence.csv"),
);

run_suite(;
    implems=IMPLEMS_MULTIPLE_SEQUENCES,
    algos=ALGOS_MULTIPLE_SEQUENCES,
    N_vals=2:2:20,
    D_vals=(1, 10),
    T_vals=100,
    K_vals=100,
    seconds=10,
    samples=100,
    path=joinpath(@__DIR__, "results", "results_multiple_sequences.csv"),
);
