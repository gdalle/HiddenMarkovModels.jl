using Revise

includet("julia/hmms.jl")
includet("julia/hmmbase.jl")
includet("julia/hmmlearn.jl")
includet("julia/pomegranate.jl")
includet("julia/julia_suite.jl")
includet("python/python_suite.jl")

function run_full_suite(;
    implems, N_vals, D_vals, T_vals, K_vals, I_vals, julia_path, python_path
)
    implems_julia = filter(implem -> contains(implem, "jl"), implems)
    implems_python = filter(implem -> !contains(implem, "jl"), implems)

    julia_results = run_julia_suite(;
        implems=implems_julia,
        N_vals=N_vals,
        D_vals=D_vals,
        T_vals=T_vals,
        K_vals=K_vals,
        I_vals=I_vals,
        seconds=1,
        path=julia_path,
    )

    python_results = run_python_suite(;
        implems=implems_python,
        N_vals=N_vals,
        D_vals=D_vals,
        T_vals=T_vals,
        K_vals=K_vals,
        I_vals=I_vals,
        samples=5,
        path=python_path,
    )

    return (; julia_results, python_results)
end
