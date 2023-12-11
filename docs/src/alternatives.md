# Competitors

## Julia

We compare features among the following Julia packages:

* HiddenMarkovModels.jl (abbreviated to HMMs.jl)
* [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl)
* [HMMGradients.jl](https://github.com/idiap/HMMGradients.jl)

We discard [MarkovModels.jl](https://github.com/FAST-ASR/MarkovModels.jl) because its focus is GPU computation.
There are also more generic packages for probabilistic programming, which are able to perform MCMC or variational inference (eg. [Turing.jl](https://github.com/TuringLang/Turing.jl)) but we leave those aside.

|                           | HMMs.jl             | HMMBase.jl          | HMMGradients.jl |
| ------------------------- | ------------------- | ------------------- | --------------- |
| Algorithms                | Sim, FB, Vit, BW    | Sim, FB, Vit, BW    | FB              |
| Observation types         | anything            | `Number` / `Vector` | anything        |
| Observation distributions | DensityInterface.jl | Distributions.jl    | manual          |
| Multiple sequences        | yes                 | no                  | yes             |
| Priors / structures       | possible            | no                  | possible        |
| Temporal dependency       | yes                 | no                  | no              |
| Control dependency        | yes                 | no                  | no              |
| Number types              | anything            | `Float64`           | `AbstractFloat` |
| Automatic differentiation | yes                 | no                  | yes             |
| Linear algebra            | yes                 | yes                 | no              |
| Logarithmic probabilities | halfway             | halfway             | yes             |

Sim = Simulation, FB = Forward-Backward, Vit = Viterbi, BW = Baum-Welch

## Python

* [hmmlearn](https://github.com/hmmlearn/hmmlearn) (based on NumPy)
* [pomegrnate](https://github.com/jmschrei/pomegranate) (based on PyTorch)
* [dynamax](https://github.com/probml/dynamax) (based on JAX)