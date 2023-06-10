using Pkg
Pkg.activate(@__DIR__)

using Revise
includet("utils/full_suite.jl")

SUITE = define_julia_suite(;
    implems=("HMMs.jl",),
    N_vals=2:2:10,
    D_vals=(1, 5),
    T_vals=500,
    K_vals=(1, 10),
    I_vals=10,
    seconds=5,
)
