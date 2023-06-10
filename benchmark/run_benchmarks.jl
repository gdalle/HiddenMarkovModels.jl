using Pkg
Pkg.activate(@__DIR__)

using Revise
includet("utils/full_suite.jl")

run_full_suite(;
    implems=(
        "HMMs.jl",
        "HMMBase.jl",
        "hmmlearn",
        "hmmlearn (jl)",
        "pomegranate",
        "pomegranate (jl)",
    ),
    N_vals=2:2:10,
    D_vals=(1, 5),
    T_vals=500,
    I_vals=10,
    K_vals=(1, 10),
    julia_path=joinpath(@__DIR__, "results", "results_julia.json"),
    python_path=joinpath(@__DIR__, "results", "results_python.json"),
);
