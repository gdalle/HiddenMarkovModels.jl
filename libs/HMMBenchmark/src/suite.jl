function to_namedtuple(x)
    return NamedTuple(n => getfield(x, n) for n in fieldnames(typeof(x)))
end

function define_suite(
    rng::AbstractRNG,
    implems::Vector{<:Implementation}=[HiddenMarkovModelsImplem()];
    instances::Vector{<:Instance},
    algos::Vector{String},
)
    SUITE = BenchmarkGroup()
    for implem in implems
        for instance in instances
            bench_tup = build_benchmarkables(rng, implem; instance, algos)
            for (algo, bench) in pairs(bench_tup)
                SUITE[string(implem)][string(instance)][algo] = bench
            end
        end
    end
    return SUITE
end

function parse_results(results; path=nothing, agg=minimum)
    results = agg(results)
    data = DataFrame()
    for implem_str in identity.(keys(results))
        for instance_str in identity.(keys(results[implem_str]))
            instance = Instance(instance_str)
            for algo in identity.(keys(results[implem_str][instance_str]))
                perf = results[implem_str][instance_str][algo]
                (; time, gctime, memory, allocs) = perf
                row = merge(
                    (; implem=implem_str, algo),
                    to_namedtuple(instance),
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
