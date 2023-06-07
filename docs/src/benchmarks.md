# Benchmarks

These benchmarks were generated during CI with the following setup:
```@repl
using InteractiveUtils
versioninfo()
```

The test case was a HMM with one-dimensional Gaussian observations, initialized randomly.
Since HiddenMarkovModels.jl and HMMBase.jl give the exact same results, the only thing to compare is their speed of execution.

![Logdensity benchmark](../../benchmark/results/Logdensity.png)
![Viterbi benchmark](../../benchmark/results/Viterbi.png)
![Forward-backward benchmark](../../benchmark/results/Forward-backward.png)
![Baum-Welch benchmark](../../benchmark/results/Baum-Welch.png)