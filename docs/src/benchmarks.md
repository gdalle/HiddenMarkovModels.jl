# Benchmarks

These benchmarks were generated with the following setup:

```@repl
using InteractiveUtils
versioninfo()
```

The test case is an HMM with multi-dimensional Gaussian observations, initialized randomly.
We compare the following packages:

- HiddenMarkovModels.jl (abbreviated to HMMs.jl)
- [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl)
- [hmmlearn](https://github.com/hmmlearn/hmmlearn)
- [pomegranate](https://github.com/jmschrei/pomegranate)

For now, pomegranate is not included on the plots because it looks much slower.
I have reached out to the lead dev to make sure my benchmark is fair.

![Logdensity benchmark](./assets/benchmark_Logdensity_T=200.svg)

![Viterbi benchmark](./assets/benchmark_Viterbi_T=200.svg)

![Forward-backward benchmark](./assets/benchmark_Forward-backward_T=200.svg)

![Baum-Welch benchmark](./assets/benchmark_Baum-Welch_T=200.svg)

The full benchmark logs are available in JSON format: [results from Julia](./assets/results_julia.json) and [results from Python](./assets/results_python.json).
Take a look at the code in `benchmark/utils` to see how they were generated.
