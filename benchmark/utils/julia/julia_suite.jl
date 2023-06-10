using BenchmarkTools

function benchmarkables_by_implem(; implem::String, kwargs...)
    if implem == "HMMs.jl"
        return benchmarkables_hmms(; kwargs...)
    elseif implem == "HMMBase.jl"
        return benchmarkables_hmmbase(; kwargs...)
    elseif implem == "hmmlearn (jl)"
        return benchmarkables_hmmlearn(; kwargs...)
    elseif implem == "pomegranate (jl)"
        return benchmarkables_pomegranate(; kwargs...)
    end
end

function define_julia_suite(; implems, N_vals, D_vals, T_vals, K_vals, I_vals)
    SUITE = BenchmarkGroup()
    if ("HMMBase.jl" in implems) && any(>(1), K_vals)
        @warn "HMMBase.jl doesn't support multiple observation sequences, concatenating instead."
    end
    for implem in implems
        SUITE[implem] = BenchmarkGroup()
        for N in N_vals, D in D_vals, T in T_vals, K in K_vals, I in I_vals
            bench_tup = benchmarkables_by_implem(; implem, N, D, T, K, I)
            for (algo, bench) in pairs(bench_tup)
                if !haskey(SUITE[implem], algo)
                    SUITE[implem][algo] = BenchmarkGroup()
                end
                SUITE[implem][algo][(N, D, T, K, I)] = bench
            end
        end
    end
    return SUITE
end

function run_julia_suite(;
    implems, N_vals, D_vals, T_vals, K_vals, I_vals, path=nothing, kwargs...
)
    @info "Julia benchmarks"
    SUITE = define_julia_suite(; implems, N_vals, D_vals, T_vals, K_vals, I_vals)
    raw_results = BenchmarkTools.run(SUITE; verbose=true, kwargs...)
    if !isnothing(path)
        BenchmarkTools.save(path, raw_results)
    end
    return raw_results
end
