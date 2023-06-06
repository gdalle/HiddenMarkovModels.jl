# HiddenMarkovModels.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/HiddenMarkovModels.jl/dev/)
[![Build Status](https://github.com/gdalle/HiddenMarkovModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gdalle/HiddenMarkovModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://app.codecov.io/gh/gdalle/HiddenMarkovModels.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/gdalle/HiddenMarkovModels.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

A Julia package for HMM modeling, simulation, inference and learning.

## Main features

- Efficiency
  - allocation-free versions of core functions
  - linear algebra subroutines
- Generic states
  - dense or sparse
  - with or without prior
- Generic observations
  - [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
  - [MeasureTheory.jl](https://github.com/cscherrer/MeasureTheory.jl)
  - anything that follows [DensityInterface.jl](https://github.com/JuliaMath/DensityInterface.jl)
- Generic number types
  - floating point precision
  - [LogarithmicNumbers.jl](https://github.com/cjdoris/LogarithmicNumbers.jl)
- Automatic differentiation of parameters
  - in forward mode with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
  - in reverse mode with [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) (WIP)

## Inspirations

- [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl)
- [HMMGradients.jl](https://github.com/idiap/HMMGradients.jl)
- [ControlledHiddenMarkovModels.jl](https://github.com/gdalle/ControlledHiddenMarkovModels.jl)
