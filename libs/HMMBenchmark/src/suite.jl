function define_suite(rng::AbstractRNG; implems, configurations, algos)
    SUITE = BenchmarkGroup()
    for implem in implems
        SUITE[implem] = BenchmarkGroup()
        for configuration in configurations
            SUITE[implem][to_tuple(configuration)] = BenchmarkGroup()
            bench_tup = benchmarkables_by_implem(rng; implem, configuration, algos)
            for (algo, bench) in pairs(bench_tup)
                SUITE[implem][to_tuple(configuration)][algo] = bench
            end
        end
    end
    return SUITE
end

function benchmarkables_by_implem(rng::AbstractRNG; implem, configuration, algos)
    if implem == "HiddenMarkovModels.jl"
        return benchmarkables_hiddenmarkovmodels(rng; configuration, algos)
    elseif implem == "HMMBase.jl"
        return benchmarkables_hmmbase(rng; configuration, algos)
    elseif implem == "hmmlearn"
        return benchmarkables_hmmlearn(rng; configuration, algos)
    elseif implem == "pomegranate"
        return benchmarkables_pomegranate(rng; configuration, algos)
    elseif implem == "dynamax"
        return benchmarkables_dynamax(rng; configuration, algos)
    else
        throw(ArgumentError("Unknown implementation"))
    end
end

function parse_results(results; path=nothing)
    data = DataFrame()
    for implem in identity.(keys(results))
        for configuration_tup in identity.(keys(results[implem]))
            configuration = Configuration(configuration_tup...)
            for algo in identity.(keys(results[implem][configuration_tup]))
                perf = results[implem][configuration_tup][algo]
                (; time, gctime, memory, allocs) = perf
                row = merge(
                    (; implem, algo),
                    to_namedtuple(configuration),
                    (; time, gctime, memory, allocs),
                )
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
