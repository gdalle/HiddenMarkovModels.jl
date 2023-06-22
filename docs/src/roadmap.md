# Roadmap

Here are some of the things that I would like to work on soon-ish:

- specification and testing for an `AbstractHMM` interface, perhaps with [Interfaces.jl](https://github.com/rafaqz/Interfaces.jl)
- numerical stability in large-dimensional settings with sparse transitions
- reverse mode autodiff with [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)
- SIMD optimization with [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl) or [Tullio.jl](https://github.com/mcabbott/Tullio.jl)
- [spectral estimation methods](https://arxiv.org/abs/0811.4413)
- [input-output HMMs](https://pubmed.ncbi.nlm.nih.gov/18263517/) in my other package [ControlledMarkovModels.jl](https://github.com/gdalle/ControlledHiddenMarkovModels.jl)

Contributors are welcome!

In the long run, I will probably transfer this package to [JuliaStats](https://github.com/JuliaStats), but for now I'd like to keep control until things are stabilized.