using BenchmarkTools
using PkgBenchmark

pkg = dirname(@__DIR__) # this git repo
baseline = "12fa976"  # commit id
target = "1b9a7f1"  # commit id

results_baseline = benchmarkpkg(pkg, baseline; verbose=true, retune=false)
results_target = benchmarkpkg(pkg, target; verbose=true, retune=false)

judgement = judge(results_baseline, results_target, minimum)

export_markdown(joinpath(@__DIR__, "benchmark_judgement.md"), judgement)
