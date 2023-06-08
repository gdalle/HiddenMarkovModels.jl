include("utils.jl")

run_full_suite(; N_values=2:2:10, D_values=3, T_values=500, max_iterations=10)

run_python_suite(; N_values=2:2:10, D_values=3, T_values=500, max_iterations=10)
