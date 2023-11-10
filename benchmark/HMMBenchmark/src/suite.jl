function benchmarkables_by_implem(; implem, algos, kwargs...)
    if implem == "HMMs.jl"
        return benchmarkables_hmms(; algos, kwargs...)
    elseif implem == "HMMBase.jl"
        return benchmarkables_hmmbase(; algos, kwargs...)
    elseif implem == "hmmlearn"
        return benchmarkables_hmmlearn(; algos, kwargs...)
    elseif implem == "pomegranate"
        return benchmarkables_pomegranate(; algos, kwargs...)
    else
        throw(ArgumentError("Unknown implementation"))
    end
end

function define_suite(; implems, algos, N_vals, D_vals, T_vals, K_vals, I)
    SUITE = BenchmarkGroup()
    if ("HMMBase.jl" in implems) && any(>(1), K_vals)
        @warn "HMMBase.jl doesn't support multiple observation sequences, concatenating instead."
    end
    for implem in implems
        SUITE[implem] = BenchmarkGroup()
        bench_tup = benchmarkables_by_implem(; implem, algos, N=1, D=1, T=2, K=1, I=1)
        for (algo, bench) in pairs(bench_tup)
            SUITE[implem][algo] = BenchmarkGroup()
            SUITE[implem][algo][(1, 1, 2, 1, 1)] = bench
        end
        for N in N_vals, D in D_vals, T in T_vals, K in K_vals
            bench_tup = benchmarkables_by_implem(; implem, algos, N, D, T, K, I)
            for (algo, bench) in pairs(bench_tup)
                SUITE[implem][algo][(N, D, T, K, I)] = bench
            end
        end
    end
    return SUITE
end

julia_implems(implems) = filter(i -> contains(i, ".jl"), implems)
python_implems(implems) = filter(i -> !contains(i, ".jl"), implems)

function run_suite(;
    implems, algos, N_vals, D_vals, T_vals, K_vals, I, path=nothing, kwargs...
)
    julia_suite = define_suite(;
        implems=julia_implems(implems), algos, N_vals, D_vals, T_vals, K_vals, I
    )
    python_suite = define_suite(;
        implems=python_implems(implems), algos, N_vals, D_vals, T_vals, K_vals, I
    )

    default_openblas_threads = BLAS.get_num_threads()

    @info "Running Python benchmarks with OPENBLAS_NUM_THREADS=$default_openblas_threads"
    raw_python_results = BenchmarkTools.run(python_suite; verbose=true, evals=1, kwargs...)

    @info "Running Julia benchmarks with OPENBLAS_NUM_THREADS=1"
    BLAS.set_num_threads(1)
    raw_julia_results = nothing
    try
        raw_julia_results = BenchmarkTools.run(
            julia_suite; verbose=true, evals=1, kwargs...
        )
    finally
        BLAS.set_num_threads(default_openblas_threads)
    end

    julia_results = minimum(raw_julia_results)
    python_results = minimum(raw_python_results)
    return (; julia_results, python_results)
end

function parse_results(many_results...; path=nothing)
    data = DataFrame()
    for results in many_results
        for implem in identity.(keys(results))
            for algo in identity.(keys(results[implem]))
                for (N, D, T, K, I) in identity.(keys(results[implem][algo]))
                    perf = results[implem][algo][(N, D, T, K, I)]
                    @unpack time, gctime, memory, allocs = perf
                    row = (; implem, algo, N, D, T, K, I, time, gctime, memory, allocs)
                    push!(data, row)
                end
            end
        end
    end

    if !isnothing(path)
        open(path, "w") do file
            CSV.write(file, data)
        end
    end
    return data
end

function print_setup(; path)
    open(path, "w") do file
        redirect_stdout(file) do
            versioninfo()
            println("\n# Multithreading\n")
            println("Julia threads = $(Threads.nthreads())")
            println("OpenBLAS threads = $(BLAS.get_num_threads())")
            println("\n# Julia packages\n")
            Pkg.status()
        end
    end
end
