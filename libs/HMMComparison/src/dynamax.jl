struct dynamaxImplem <: Implementation end

function HMMBenchmark.build_benchmarkables(
    rng::AbstractRNG, implem::dynamaxImplem; instance::Instance, algos::Vector{String}
)
    benchs = Dict()
    return benchs
end
