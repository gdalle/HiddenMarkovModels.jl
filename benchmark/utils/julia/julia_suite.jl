using BenchmarkTools

function benchmarkables(; implem::String, N, D, T, I)
    if implem == "HMMs.jl"
        return benchmarkables_hmms(; N, D, T, I)
    elseif implem == "HMMBase.jl"
        return benchmarkables_hmmbase(; N, D, T, I)
    end
end

function define_julia_suite(; implems, N_values, D_values, T_values, I_values)
    SUITE = BenchmarkGroup()
    for implem in implems
        SUITE[implem] = BenchmarkGroup()
        for N in N_values, D in D_values, T in T_values, I in I_values
            SUITE[implem][(N, D, T, I)] = BenchmarkGroup()
            bench_tup = benchmarkables(; implem, N, D, T, I)
            for (name, bench) in pairs(bench_tup)
                SUITE[implem][(N, D, T, I)][name] = bench
            end
        end
    end
    return SUITE
end

function run_julia_suite(;
    implems, N_values, D_values, T_values, I_values, samples, path=nothing
)
    @info "Julia benchmarks"
    SUITE = define_julia_suite(; implems, N_values, D_values, T_values, I_values)
    raw_results = BenchmarkTools.run(SUITE; verbose=true, samples=samples)
    if !isnothing(path)
        BenchmarkTools.save(path, raw_results)
    end
    return raw_results
end
