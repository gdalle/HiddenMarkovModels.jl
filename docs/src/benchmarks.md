# Benchmarks

These benchmarks were generated with the following setup:
```@repl
using InteractiveUtils
versioninfo()
```

The test case was a HMM with one-dimensional Gaussian observations, initialized randomly.
Since HiddenMarkovModels.jl and HMMBase.jl give the exact same results, the only thing to compare is their speed of execution.

You can check out the complete benchmarking results in this [JSON file](assets/benchmark_results.json) created by BenchmarkTools.jl.

![Logdensity benchmark](assets/benchmark_Logdensity.png)

![Viterbi benchmark](assets/benchmark_Viterbi.png)

![Forward-backward benchmark](assets/benchmark_Forward-backward.png)

![Baum-Welch benchmark](assets/benchmark_Baum-Welch.png)
