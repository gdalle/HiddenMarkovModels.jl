function define_suite(rng::AbstractRNG; configurations, algos)
    SUITE = BenchmarkGroup()
    implem = "HiddenMarkovModels.jl"
    for configuration in configurations
        bench_tup = benchmarkables_hiddenmarkovmodels(rng; configuration, algos)
        for (algo, bench) in pairs(bench_tup)
            SUITE[implem][string(configuration)][algo] = bench
        end
    end
    return SUITE
end

function parse_results(results; path=nothing)
    data = DataFrame()
    for implem in identity.(keys(results))
        for configuration_str in identity.(keys(results[implem]))
            configuration = Configuration(configuration_str)
            for algo in identity.(keys(results[implem][configuration_str]))
                perf = results[implem][configuration_str][algo]
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

function print_julia_setup(; path)
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
