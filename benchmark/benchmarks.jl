include("utils/julia_suite.jl")

SUITE = define_julia_suite(;
    N_values=2:2:10,
    D_values=3,
    T_values=100,
    I=5,
    include_python=false,
    include_hmmbase=false,
)
