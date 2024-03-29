# Debugging

## Numerical underflow

The most frequent error you will encounter is an underflow during inference, caused by some values being infinite or `NaN`.
This can happen for a variety of reasons, so here are a few leads worth investigating:

* Increase the duration of the sequence / the number of sequences to get more data
* Add a prior to your transition matrix / observation distributions to avoid degenerate behavior (like zero variance in a Gaussian or zero probability in a Bernoulli)
* Reduce the number of states to make every one of them useful
* Pick a better initialization to start closer to the supposed ground truth
* Use numerically stable number types (such as [LogarithmicNumbers.jl](https://github.com/cjdoris/LogarithmicNumbers.jl)) in strategic places, but beware: these numbers don't play nicely with Distributions.jl, so you may have to roll out your own [Custom distributions](@ref).

## Method errors

This might be caused by: 

* forgetting to define methods for your custom type
* omitting `control_seq` or `seq_ends` in some places.

Check the [API reference](@ref).

## Performance

If your algorithms are too slow, you can leverage the existing [Interfaces](@ref) to improve the components of your model separately (first observation distributions, then fitting).
The usual advice always applies:

* Use [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) to establish a baseline
* Use profiling to see where you spend most of your time
* Use [JET.jl](https://github.com/aviatesk/JET.jl) to track down type instabilities
* Use [AllocCheck.jl](https://github.com/JuliaLang/AllocCheck.jl) to reduce allocations
