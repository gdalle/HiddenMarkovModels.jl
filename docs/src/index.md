# HiddenMarkovModels.jl

A Julia package for HMM modeling, simulation, inference and learning.[^1]

[^1]: Logo by Clément Mantoux

## Main features

!!! info "Performance"
    - allocation-free versions of core functions
    - linear algebra subroutines
    - multithreading
    - compatibility with [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)

!!! info "Genericity"
    - transition matrices can be dense or sparse
    - observations can be arbitrary Julia objects (not just numbers or arrays)
    - emission distributions `d` must only implement `rand(d)` and `logdensityof(d, x)` (as per [DensityInterface.jl](https://github.com/JuliaMath/DensityInterface.jl))
    - possibility to add priors
    - number types are not restricted

!!! info "Automatic differentiation"
    - in forward mode with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
    - in reverse mode with [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) (WIP)
