# Competitors

## Julia

We compare features among the following Julia packages:

* HiddenMarkovModels.jl (abbreviated to HMMs.jl)
* [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl)
* [HMMGradients.jl](https://github.com/idiap/HMMGradients.jl)

We discard [MarkovModels.jl](https://github.com/FAST-ASR/MarkovModels.jl) because its focus is GPU computation.
There are also more generic packages for probabilistic programming, which are able to perform MCMC or variational inference (eg. [Turing.jl](https://github.com/TuringLang/Turing.jl)) but we leave those aside.

|                           | HMMs.jl             | HMMBase.jl       | HMMGradients.jl |
| ------------------------- | ------------------- | ---------------- | --------------- |
| Algorithms[^1]            | V, FB, BW           | V, FB, BW        | FB              |
| Number types              | anything            | `Float64`        | `AbstractFloat` |
| Observation types         | anything            | number or vector | anything        |
| Observation distributions | DensityInterface.jl | Distributions.jl | manual          |
| Multiple sequences        | yes                 | no               | yes             |
| Priors / structures       | possible            | no               | possible        |
| Temporal dependency       | yes                 | no               | no              |
| Control dependency        | yes                 | no               | no              |
| Automatic differentiation | yes                 | no               | yes             |
| Linear algebra speedup    | yes                 | yes              | no              |
| Numerical stability       | scaling+            | scaling+         | log             |


!!! info "Very small probabilities"
    In all HMM algorithms, we work with probabilities that may become very small as time progresses.
    There are two main solutions for this problem: scaling and logarithmic computations.
    This package implements the Viterbi algorithm in log scale, but the other algorithms use scaling to exploit BLAS operations.
    As was done in HMMBase.jl, we enhance scaling with a division by the highest observation loglikelihood: instead of working with $b_{i,t} = \mathbb{P}(Y_t | X_t = i)$, we use $b_{i,t} / \max_i b_{i,t}$.
    See [Formulas](@ref) for details.

## Python

We compare features among the following Python packages:

* [hmmlearn](https://github.com/hmmlearn/hmmlearn) (based on NumPy)
* [pomegranate](https://github.com/jmschrei/pomegranate) (based on PyTorch)
* [dynamax](https://github.com/probml/dynamax) (based on JAX)

|                           | hmmlearn             | pomegranate           | dynamax              |
| ------------------------- | -------------------- | --------------------- | -------------------- |
| Algorithms[^1]            | V, FB, BW, VI        | V, FB, BW             | FB, V, BW, GD        |
| Number types              | NumPy format         | PyTorch format        | JAX format           |
| Observation types         | number or vector     | number or vector      | number or vector     |
| Observation distributions | discrete or Gaussian | pomegranate catalogue | discrete or Gaussian |
| Multiple sequences        | yes                  | yes                   | yes                  |
| Priors / structures       | yes                  | no                    | ?                    |
| Temporal dependency       | no                   | no                    | no                   |
| Control dependency        | no                   | no                    | no                   |
| Automatic differentiation | no                   | yes                   | yes                  |
| Linear algebra speedup    | yes                  | yes                   | yes                  |
| Logarithmic probabilities | scaling / log        | log                   | log                  |


[^1]: V = Viterbi, FB = Forward-Backward, BW = Baum-Welch, VI = Variational Inference, GD = Gradient Descent
