using Pkg
Pkg.activate(@__DIR__)

using Revise
includet("utils/full_suite.jl")

SUITE = define_julia_suite(;
    implems=("HMMs.jl",), N_values=2:2:10, D_values=(1, 5), T_values=200, I_values=5
)
