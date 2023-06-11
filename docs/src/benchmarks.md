# Benchmarks

The test case is an HMM with diagonal multivariate normal observations.
We compare the following packages:

- HiddenMarkovModels.jl (abbreviated to HMMs.jl)
- [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl)
- [hmmlearn](https://github.com/hmmlearn/hmmlearn)
- [pomegranate](https://github.com/jmschrei/pomegranate)


## Results

### Single sequence

Full benchmark logs: [`results_single_sequence.csv`](./assets/benchmark/results/results_single_sequence.csv).

![Logdensity single sequence benchmark](./assets/benchmark/plots/benchmark_single_sequence_logdensity.svg)

![Viterbi single sequence benchmark](./assets/benchmark/plots/benchmark_single_sequence_viterbi.svg)

![Forward-backward single sequence benchmark](./assets/benchmark/plots/benchmark_single_sequence_forward_backward.svg)

![Baum-Welch single sequence benchmark](./assets/benchmark/plots/benchmark_single_sequence_baum_welch.svg)

Here, pomegranate is not included because it is much slower on very small inputs.

### Multiple sequences

Full benchmark logs: [`results_multiple_sequences.csv`](./assets/benchmark/results/results_multiple_sequences.csv).

![Logdensity single sequence benchmark](./assets/benchmark/plots/benchmark_multiple_sequences_logdensity.svg)

![Baum-Welch single sequence benchmark](./assets/benchmark/plots/benchmark_multiple_sequences_baum_welch.svg)

Here, HMMBase.jl is not included because it does not support multiple sequences.

## Reproducibility

These benchmarks were generated in the following environment: [`setup.txt`](./assets/benchmark/results/setup.txt).

If you want to run them on your machine:

1. Clone the [HiddenMarkovModels.jl](https://github.com/gdalle/HiddenMarkovModels.jl) repository
2. Open a Julia REPL at the root
3. Run the following commands

   ```julia
   include("benchmark/run_benchmarks.jl")
   include("docs/process_benchmarks.jl")
   ```

## Remarks

### Allocations

A major bottleneck of performance in Julia is memory allocations.
The benchmarks for HMMs.jl thus employ a custom implementation of diagonal multivariate normals, which is entirely allocation-free.

This partly explains the performance gap with HMMBase.jl as the dimension `D` grows beyond 1.
Such a trick is also possible with HMMBase.jl, but slightly more demanding since it requires subtyping `Distribution` from Distributions.jl, instead of just implementing DensityInterface.jl.
We might do it in future benchmarks.

### Parallelism

The packages we include have different approaches to parallelism, which can bias the evaluation in complex ways:

| Package    | States `N`        | Observations `D` | Sequences `K` |
| ---------- | ----------------- | ---------------- | ---------------- |
| HMMs.jl    | LinearAlgebra[^2] | depends[^2]      | Threads[^1]      |
| HMMBase.jl | -                 | depends[^2]      | -                |
| hmmlearn   | NumPy[^2]         | NumPy[^2]        | NumPy[^2]        |
| hmmlearn   | PyTorch[^3]       | PyTorch[^3]      | PyTorch[^3]      |

[^1]: possibly affected by `JULIA_NUM_THREADS`
[^2]: possibly affected by `OPENBLAS_NUM_THREADS`
[^3]: possibly affected by `MKL_NUM_THREADS`

For a fairer comparison, we set `JULIA_NUM_THREADS=1`, even though it robs HMMs.jl of its parallel speedup on multiple sequences.

In addition, OpenBLAS threads have [negative interactions](https://github.com/JuliaLang/julia/issues/44201#issuecomment-1585656581) with Julia threads.
To overcome this obstacle, we run the Julia benchmarks (and only those) with `OPENBLAS_NUM_THREADS=1`.

## Acknowledgements

A big thank you to [Maxime Mouchet](https://www.maxmouchet.com/) and [Jacob Schreiber](https://jmschrei.github.io/), the respective lead devs of HMMBase.jl and pomegranate, for their help.
