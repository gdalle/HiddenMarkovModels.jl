# Alternatives

## List

For exact inference in HMMs, we have identified the following Julia packages:

* [HMMBase.jl]
* [HMMGradients.jl]
* [MarkovModels.jl]

and the following Python packages:

* [hmmlearn]
* [pomegranate]

There are also more generic packages for probabilistic programming, which are able to perform MCMC or variational inference (eg. [Turing.jl]) but we leave those aside.

## Features

We only compare features among Julia packages.
If you are interested in performance, the benchmarks page also includes Python packages.

|                           | HiddenMarkovModels.jl | HMMBase.jl          | HMMGradients.jl |
| ------------------------- | --------------------- | ------------------- | --------------- |
| Algorithms                | Sim, FB, Vit, BW      | Sim, FB, Vit, BW    | FB              |
| Observation types         | anything              | `Number` / `Vector` | anything        |
| Observation distributions | [DensityInterface.jl] | [Distributions.jl]  | manual          |
| Number types              | anything              | `Float64`           | `AbstractFloat` |
| Priors / structures       | possible              | no                  | possible        |
| Automatic differentiation | yes                   | no                  | yes             |
| Multiple sequences        | yes                   | no                  | yes             |
| Linear algebra            | yes                   | yes                 | no              |
| GPU support               | ?                     | ?                   | ?               |

Sim = Simulation, FB = Forward-Backward, Vit = Viterbi, BW = Baum-Welch

<!-- Links -->

[hmmlearn]: https://github.com/hmmlearn/hmmlearn
[pomegranate]: https://github.com/jmschrei/pomegranate

[DensityInterface.jl]: https://github.com/JuliaMath/DensityInterface.jl
[Distributions.jl]: https://github.com/JuliaStats/Distributions.jl
[HMMBase.jl]: https://github.com/maxmouchet/HMMBase.jl
[HMMGradients.jl]: https://github.com/idiap/HMMGradients.jl
[MarkovModels.jl]: https://github.com/FAST-ASR/MarkovModels.jl
[Turing.jl]: https://github.com/TuringLang/Turing.jl