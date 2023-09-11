# HiddenMarkovModels.jl

<img alt="HiddenMarkovModels logo" src="docs/src/assets/logo.png" width="150" height="150" align="right" />

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gdalle.github.io/HiddenMarkovModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/HiddenMarkovModels.jl/dev/)
[![Build Status](https://github.com/gdalle/HiddenMarkovModels.jl/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/gdalle/HiddenMarkovModels.jl/actions/workflows/test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/gdalle/HiddenMarkovModels.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/gdalle/HiddenMarkovModels.jl)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8128331.svg)](https://doi.org/10.5281/zenodo.8128331)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red)](https://github.com/aviatesk/JET.jl)

A Julia package for HMM modeling, simulation, inference and learning.

## Getting started

This package can be installed using Julia's package manager:

```julia
pkg> add HiddenMarkovModels
```

Then, you can create your first HMM as follows:

```julia
using Distributions, HiddenMarkovModels
init = [0.2, 0.8]
trans = [0.1 0.9; 0.7 0.3]
dists = [Normal(-1), Normal(1)]
hmm = HMM(init, trans, dists)
```

Take a look at the [documentation](https://gdalle.github.io/HiddenMarkovModels.jl/stable/) to know what to do next!

## Main features

#### Genericity

- observations can be arbitrary Julia objects, not just scalars or arrays
- number types are not restricted
- emission distributions only need to implement [DensityInterface.jl](https://github.com/JuliaMath/DensityInterface.jl): `rand(rng, dist)` and `logdensityof(dist, x)`

#### Performance

- allocation-free versions of core functions
- leveraging linear algebra subroutines and multithreading
- compatibility with [SparseArrays.jl](https://github.com/JuliaSparse/SparseArrays.jl) and [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)

#### Reliability

- same outputs as [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl), up to numerical precision
- quality checks with [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) and type stability checks with [JET.jl](https://github.com/aviatesk/JET.jl)
- documentation with [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) and benchmarks with [PkgBenchmark.jl](https://github.com/JuliaCI/PkgBenchmark.jl)

#### Automatic differentiation

- in forward mode with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
- in reverse mode with [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)

## Alternatives

| Julia                                                                                                                                                                               | Python                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl) <br> [MarkovModels.jl](https://github.com/FAST-ASR/MarkovModels.jl) <br> [HMMGradients.jl](https://github.com/idiap/HMMGradients.jl) | [hmmlearn](https://github.com/hmmlearn/hmmlearn) <br> [pomegranate](https://github.com/jmschrei/pomegranate) |

## Contributing

If you spot a bug or want to ask about a new feature, please [open an issue](https://github.com/gdalle/HiddenMarkovModels.jl/issues) on the GitHub repository.

## Acknowledgements

A big thank you to [Maxime Mouchet](https://www.maxmouchet.com/) and [Jacob Schreiber](https://jmschrei.github.io/), the respective lead devs of [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl) and [pomegranate](https://github.com/jmschrei/pomegranate), for their help and advice.

Logo by [Clément Mantoux](https://cmantoux.github.io/) based on a portrait of [Andrey Markov](https://en.wikipedia.org/wiki/Andrey_Markov).
