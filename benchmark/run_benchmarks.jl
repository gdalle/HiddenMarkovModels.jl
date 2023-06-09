using Revise

includet("utils/julia_suite.jl")
includet("utils/python_suite.jl")

function run_full_suite(; N_values, D_values, T_values, I, julia_path, python_path)
    julia_results = run_julia_suite(;
        N_values,
        D_values,
        T_values,
        I,
        include_python=false,
        include_hmmbase=true,
        seconds=5,
        path=julia_path,
    )
    python_results = run_python_suite(;
        N_values, D_values, T_values, I, number=10, repeat=10, path=python_path
    )
    return (; julia_results, python_results)
end

JULIA_RESULTS_PATH = joinpath(@__DIR__, "results", "results_julia.json")
PYTHON_RESULTS_PATH = joinpath(@__DIR__, "results", "results_python.json")

(; julia_results, python_results) = run_full_suite(;
    N_values=2:2:10,
    D_values=(1, 5),
    T_values=200,
    I=5,
    julia_path=JULIA_RESULTS_PATH,
    python_path=PYTHON_RESULTS_PATH,
)
