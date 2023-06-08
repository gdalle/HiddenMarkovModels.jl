# Benchmarks

These benchmarks were generated with the following setup:
```@repl
using InteractiveUtils
versioninfo()
```

The test case was a HMM with multi-dimensional Gaussian observations, initialized randomly.
Since HiddenMarkovModels.jl and HMMBase.jl give the exact same results, the only thing to compare is their speed of execution.

You can check out the complete benchmarking results in this [JSON file](./assets/benchmark_results.json) created by BenchmarkTools.jl.
The associated code is in `benchmark/run_benchmarks.jl`.

!!! danger "Warning"
    The benchmarks for hmmlearn are done by calling Python from Julia, which incurs an overhead that is difficult to quantify. I did my best to reduce it but I cannot guarantee it is absent. I would welcome any help investigating this!

![Logdensity benchmark](./assets/benchmark_Logdensity_D=3_T=100.png)

![Viterbi benchmark](./assets/benchmark_Viterbi_D=3_T=100.png)

![Forward-backward benchmark](./assets/benchmark_Forward-backward_D=3_T=100.png)

![Baum-Welch benchmark](./assets/benchmark_Baum-Welch_D=3_T=100.png)
