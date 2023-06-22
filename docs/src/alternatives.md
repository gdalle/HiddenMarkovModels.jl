# Alternatives

## Julia

- [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl)
- [HMMGradients.jl](https://github.com/idiap/HMMGradients.jl)
- [MarkovModels.jl](https://github.com/FAST-ASR/MarkovModels.jl)
- [ControlledHiddenMarkovModels.jl](https://github.com/gdalle/ControlledHiddenMarkovModels.jl)

## Python

- [hmmlearn](https://github.com/hmmlearn/hmmlearn)
- [pomegranate](https://github.com/jmschrei/pomegranate)

## Comparison

Here is a feature comparison between HMMBase.jl and HiddenMarkovModels.jl:

|                           | HMMBase.jl                                                              | HiddenMarkovModels.jl                                                              |
| ------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Number types              | `Float64`                                                               | anything                                                                           |
| Observation types         | `Number` or `Vector`                                                    | anything                                                                           |
| Observation distributions | from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) | satisfying [DensityInterface.jl](https://github.com/JuliaMath/DensityInterface.jl) |
| Priors / structures       | no                                                                      | customizable                                                                       |
| Autodiff                  | no                                                                      | forward mode (for now)                                                             |
| Multiple sequences        | no                                                                      | yes                                                                                |