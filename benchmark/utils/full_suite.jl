using Revise

includet("julia/hmms.jl")
includet("julia/hmmbase.jl")
includet("julia/julia_suite.jl")

includet("python/python_suite.jl")

function run_full_suite(;
    implems,
    N_values,
    D_values,
    T_values,
    I_values,
    julia_path,
    python_path,
    number=5,
    repeat=5,
    seconds=1,
)
    implems_julia = filter(implem -> endswith(implem, ".jl"), implems)
    implems_python = filter(implem -> !endswith(implem, ".jl"), implems)

    julia_results = run_julia_suite(;
        implems=implems_julia,
        N_values=N_values,
        D_values=D_values,
        T_values=T_values,
        I_values=I_values,
        seconds=seconds,
        path=julia_path,
    )

    python_results = run_python_suite(;
        implems=implems_python,
        N_values=N_values,
        D_values=D_values,
        T_values=T_values,
        I_values=I_values,
        number=number,
        repeat=repeat,
        path=python_path,
    )
    return (; julia_results, python_results)
end
