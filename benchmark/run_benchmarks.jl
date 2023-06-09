using Pkg
Pkg.activate(@__DIR__)

using Revise
includet("utils/full_suite.jl")

run_full_suite(;
    implems=("HMMs.jl", "HMMBase.jl", "hmmlearn"),
    N_values=2:2:10,
    D_values=(1, 5),
    T_values=500,
    I_values=10,
    samples=5,
    julia_path=joinpath(@__DIR__, "results", "results_julia.json"),
    python_path=joinpath(@__DIR__, "results", "results_python.json"),
)
