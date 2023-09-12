# HiddenMarkovModels.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gdalle.github.io/HiddenMarkovModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/HiddenMarkovModels.jl/dev/)
[![Build Status](https://github.com/gdalle/HiddenMarkovModels.jl/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/gdalle/HiddenMarkovModels.jl/actions/workflows/test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/gdalle/HiddenMarkovModels.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/gdalle/HiddenMarkovModels.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8128331.svg)](https://doi.org/10.5281/zenodo.8128331)
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

This package is **generic**.
Observations can be arbitrary Julia objects, not just scalars or arrays.
Their distributions only need to implement `rand(rng, dist)` and `logdensityof(dist, x)` from [DensityInterface.jl](https://github.com/JuliaMath/DensityInterface.jl).
Number types are not restricted to floating point, and automatic differentiation is supported in forward mode ([ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)) and reverse mode ([ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)).

This package is **fast**.
All the inference functions have allocation-free versions, which leverage efficient linear algebra subroutines.
Multithreading is used to parallelize computations across sequences, and compatibility with various array types ([SparseArrays.jl](https://github.com/JuliaSparse/SparseArrays.jl) and [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)) is ensured.
We include extensive benchmarks against Julia and Python competitors thanks to [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) and [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl).

This package is **reliable**.
It gives the same results as the previous reference package [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl) up to numerical accuracy.
The test suite incorporates quality checks with [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl), as well as linting and type stability checks with [JET.jl](https://github.com/aviatesk/JET.jl).
A detailed documentation will help you find the functions you need.

But this package is **limited in scope**.
It is designed for HMMs with a small number of states, because memory and runtime scale quadratically (even if the transitions are sparse).
It is also meant to perform best on a CPU, and not tested at all on GPUs.

## Contributing

If you spot a bug or want to ask about a new feature, please [open an issue](https://github.com/gdalle/HiddenMarkovModels.jl/issues) on the GitHub repository.
Once the issue receives positive feedback, feel free to try and fix it with a pull request that follows the [BlueStyle](https://github.com/invenia/BlueStyle) guidelines.

## Acknowledgements

A big thank you to [Maxime Mouchet](https://www.maxmouchet.com/) and [Jacob Schreiber](https://jmschrei.github.io/), the respective lead devs of [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl) and [pomegranate](https://github.com/jmschrei/pomegranate), for their help and advice.
Logo by [Clément Mantoux](https://cmantoux.github.io/) based on a portrait of [Andrey Markov](https://en.wikipedia.org/wiki/Andrey_Markov).
