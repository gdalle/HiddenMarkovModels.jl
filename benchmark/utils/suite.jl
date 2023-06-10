using BenchmarkTools
using CondaPkg
using CSV
using DataFrames
using LinearAlgebra
using Pkg
using Revise

BAUM_WELCH_ITER = 10

includet("hmms.jl")
includet("hmmbase.jl")
includet("hmmlearn.jl")
includet("pomegranate.jl")

function benchmarkables_by_implem(; implem::String, kwargs...)
    if implem == "HMMs.jl"
        return benchmarkables_hmms(; kwargs...)
    elseif implem == "HMMBase.jl"
        return benchmarkables_hmmbase(; kwargs...)
    elseif implem == "hmmlearn"
        return benchmarkables_hmmlearn(; kwargs...)
    elseif implem == "pomegranate"
        return benchmarkables_pomegranate(; kwargs...)
    else
        throw(ArgumentError("Unknown implementation"))
    end
end

function define_suite(; implems, N_vals, D_vals, T_vals, K_vals)
    SUITE = BenchmarkGroup()
    if ("HMMBase.jl" in implems) && any(>(1), K_vals)
        @warn "HMMBase.jl doesn't support multiple observation sequences, concatenating instead."
    end
    for implem in implems
        SUITE[implem] = BenchmarkGroup()
        for N in N_vals, D in D_vals, T in T_vals, K in K_vals
            bench_tup = benchmarkables_by_implem(; implem, N, D, T, K)
            for (algo, bench) in pairs(bench_tup)
                if !haskey(SUITE[implem], algo)
                    SUITE[implem][algo] = BenchmarkGroup()
                end
                SUITE[implem][algo][(N, D, T, K)] = bench
            end
        end
    end
    return SUITE
end

function run_suite(; implems, N_vals, D_vals, T_vals, K_vals, path=nothing, kwargs...)
    SUITE = define_suite(; implems, N_vals, D_vals, T_vals, K_vals)
    raw_results = BenchmarkTools.run(SUITE; verbose=true, kwargs...)
    results = minimum(raw_results)  # min aggregation

    data = DataFrame()
    for implem in identity.(keys(results))
        for algo in identity.(keys(results[implem]))
            for (N, D, T, K) in identity.(keys(results[implem][algo]))
                I = BAUM_WELCH_ITER
                perf = results[implem][algo][(N, D, T, K)]
                (; time, gctime, memory, allocs) = perf
                row = (; implem, algo, N, D, T, K, I, time, gctime, memory, allocs)
                push!(data, row)
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
    mkl_num_threads = get(ENV, "MKL_NUM_THREADS", "?")
    open(path, "w") do file
        redirect_stdout(file) do
            versioninfo()
            println("\n# Multithreading\n")
            println("JULIA_NUM_THREADS = $(Threads.nthreads())")
            println("OPENBLAS_NUM_THREADS = $(BLAS.get_num_threads())")
            println("MKL_NUM_THREADS = $mkl_num_threads")
            println("\n# Julia packages\n")
            Pkg.status()
            println("\n# Python packages\n")
        end
        redirect_stderr(file) do
            CondaPkg.status()
        end
    end
end
