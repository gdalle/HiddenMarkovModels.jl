include("utils.jl")

SUITE = define_suite(;
    N_values=2:2:10, D_values=1:2:5, T_values=100, max_iterations=10; include_python=false
)
