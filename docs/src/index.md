# HiddenMarkovModels.jl

A Julia package for HMM modeling, simulation, inference and learning.

## Main features

!!! info "Performance"
    - allocation-free versions of core functions
    - linear algebra subroutines
    - multithreading
    - compatibility with [SparseArrays.jl](https://github.com/JuliaSparse/SparseArrays.jl) and [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)

!!! info "Genericity"
    - number types are not restricted
    - observations can be arbitrary Julia objects (not just scalars or arrays)
    - emission distributions `dist` only need to implement
      - `rand(rng, dist)`
      - `logdensityof(dist, x)` (following [DensityInterface.jl](https://github.com/JuliaMath/DensityInterface.jl))

!!! info "Automatic differentiation"
    - in forward mode with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
    - in reverse mode with [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)

!!! info "Reliability"
    - same outputs as [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl), up to numerical precision
    - quality checks with [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl)
    - type stability checks with [JET.jl](https://github.com/aviatesk/JET.jl)
    - documentation with [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl)
    - benchmarks with [PkgBenchmark.jl](https://github.com/JuliaCI/PkgBenchmark.jl)

## Acknowledgements

A big thank you to [Maxime Mouchet](https://www.maxmouchet.com/) and [Jacob Schreiber](https://jmschrei.github.io/), the respective lead devs of [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl) and [pomegranate](https://github.com/jmschrei/pomegranate), for their help and advice.

Logo by [Clément Mantoux](https://cmantoux.github.io/) based on a portrait of [Andrey Markov](https://en.wikipedia.org/wiki/Andrey_Markov).