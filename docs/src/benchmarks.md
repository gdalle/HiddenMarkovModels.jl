# Benchmarks

These benchmarks were generated with the following setup:
```@repl
using InteractiveUtils
versioninfo()
```

The test case was a HMM with one-dimensional Gaussian observations, initialized randomly.
Since HiddenMarkovModels.jl and HMMBase.jl give the exact same results, the only thing to compare is their speed of execution.

![Logdensity benchmark](../assets/Logdensity.png)
![Viterbi benchmark](../assets/Viterbi.png)
![Forward-backward benchmark](../assets/Forward-backward.png)
![Baum-Welch benchmark](../assets/Baum-Welch.png)

!!! warning "If you don't see any plots"
    It's because benchmarking doesn't work on CI for now. Stay tuned!
