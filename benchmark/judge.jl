using Pkg
Pkg.activate(@__DIR__)

using BenchmarkTools
using PkgBenchmark

pkg = dirname(@__DIR__) # this git repo
baseline = "04ee332"  # commit id
target = "859c7b0"  # commit id

results_baseline = benchmarkpkg(pkg, baseline; verbose=true, retune=false)
results_target = benchmarkpkg(pkg, target; verbose=true, retune=false)

judgement = judge(results_baseline, results_target, minimum)

export_markdown(joinpath(@__DIR__, "benchmark_judgement.md"), judgement)
