# Benchmarks

These benchmarks were generated with the following setup:
```@repl
using InteractiveUtils
versioninfo()
```

The test case was a HMM with multi-dimensional Gaussian observations, initialized randomly.
Since HiddenMarkovModels.jl and [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl) give the exact same results, the only thing to compare is their speed of execution. We also compare them agains the Python library [hmmlearn](https://github.com/hmmlearn/hmmlearn).

![Logdensity benchmark](./assets/benchmark_Logdensity_D=3_T=500.png)

![Viterbi benchmark](./assets/benchmark_Viterbi_D=3_T=500.png)

![Forward-backward benchmark](./assets/benchmark_Forward-backward_D=3_T=500.png)

![Baum-Welch benchmark](./assets/benchmark_Baum-Welch_D=3_T=500.png)
