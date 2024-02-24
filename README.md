# HiddenMarkovModels.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gdalle.github.io/HiddenMarkovModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/HiddenMarkovModels.jl/dev/)
[![Build Status](https://github.com/gdalle/HiddenMarkovModels.jl/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/gdalle/HiddenMarkovModels.jl/actions/workflows/test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/gdalle/HiddenMarkovModels.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/gdalle/HiddenMarkovModels.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8128331.svg)](https://doi.org/10.5281/zenodo.8128331)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red)](https://github.com/aviatesk/JET.jl)

A Julia package for simulation, inference and learning of Hidden Markov Models.

## Getting started

This package can be installed using Julia's package manager:

```julia
pkg> add HiddenMarkovModels
```

Then, you can create your first model as follows:

```julia
using Distributions, HiddenMarkovModels
init = [0.6, 0.4]
trans = [0.7 0.3; 0.2 0.8]
dists = [Normal(-1.0), Normal(1.0)]
hmm = HMM(init, trans, dists)
```

Take a look at the [documentation](https://gdalle.github.io/HiddenMarkovModels.jl/stable/) to know what to do next!

## Some background

[Hidden Markov Models](https://en.wikipedia.org/wiki/Hidden_Markov_model) (HMMs) are a widely used modeling framework in signal processing, bioinformatics and plenty of other fields.
They explain an observation sequence $(Y_t)$ by assuming the existence of a latent Markovian state sequence $(X_t)$ whose current value determines the distribution of observations.
In some scenarios, the state and the observation sequence are also allowed to depend on a known control sequence $(U_t)$.
Each of the problems below has an efficient solution algorithm, available here:

| Problem    | Goal                                   | Algorithm        |
| ---------- | -------------------------------------- | ---------------- |
| Evaluation | Likelihood of the observation sequence | Forward          |
| Filtering  | Last state marginals                   | Forward          |
| Smoothing  | All state marginals                    | Forward-backward |
| Decoding   | Most likely state sequence             | Viterbi          |
| Learning   | Maximum likelihood parameter           | Baum-Welch       |

Take a look at this tutorial to know more about the math:

> [_A tutorial on hidden Markov models and selected applications in speech recognition_](https://ieeexplore.ieee.org/document/18626), Rabiner (1989)

## Main features

This package is **generic**.
Observations can be arbitrary Julia objects, not just scalars or arrays.
Number types are not restricted to floating point, which enables automatic differentiation.
Time-dependent or controlled HMMs are supported out of the box.

This package is **fast**.
All the inference functions have allocation-free versions, which leverage efficient linear algebra subroutines.
We will include extensive benchmarks against Julia and Python competitors.

This package is **reliable**.
It gives the same results as the previous reference package up to numerical accuracy.
The test suite incorporates quality checks as well as type stability and allocation analysis.

## Contributing

If you spot a bug or want to ask about a new feature, please [open an issue](https://github.com/gdalle/HiddenMarkovModels.jl/issues) on the GitHub repository.
Once the issue receives positive feedback, feel free to try and fix it with a pull request that follows the [BlueStyle](https://github.com/invenia/BlueStyle) guidelines.

## Acknowledgements

A big thank you to [Maxime Mouchet](https://www.maxmouchet.com/) and [Jacob Schreiber](https://jmschrei.github.io/), the respective lead devs of alternative packages [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl) and [pomegranate](https://github.com/jmschrei/pomegranate), for their help and advice.
Logo by [Clément Mantoux](https://cmantoux.github.io/) based on a portrait of [Andrey Markov](https://en.wikipedia.org/wiki/Andrey_Markov).
