include("utils.jl")

const SUITE = define_suite(; N_values=2:2:10, T=100, baum_welch_iterations=10)
