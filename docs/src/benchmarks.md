# Benchmarks

These benchmarks were generated with the following setup:
```@repl
using InteractiveUtils
versioninfo()
```

!!! warning "Work in progress"
    In the meantime, you can check out the latest [raw results in JSON](benchmarks.json) and analyze them with [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl):
    ```julia
    using BenchmarkTools
    results = BenchmarkTools.read("benchmarks.json")
    ```