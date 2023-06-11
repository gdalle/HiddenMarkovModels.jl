# Benchmarks

The test case is an HMM with diagonal multivariate normal observations.
We compare the following packages:

- HiddenMarkovModels.jl (abbreviated to HMMs.jl)
- [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl)
- [hmmlearn](https://github.com/hmmlearn/hmmlearn)
- [pomegranate](https://github.com/jmschrei/pomegranate)

Since HMMBase.jl does not support multiple trajectories, we concatenate them instead.
For now, pomegranate is not included on the plots because it is much slower on very small inputs.

## Results

![Logdensity benchmark](./assets/benchmark_logdensity.svg)

![Viterbi benchmark](./assets/benchmark_viterbi.svg)

![Forward-backward benchmark](./assets/benchmark_forward_backward.svg)

![Baum-Welch benchmark](./assets/benchmark_baum_welch.svg)

The full benchmark logs are available in CSV format: [`results.csv`](./assets/results.csv).

## Reproducibility

These benchmarks were generated in the following environment: [`setup.txt`](./assets/setup.txt).

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

| Package    | States `N`        | Observations `D` | Trajectories `K` |
| ---------- | ----------------- | ---------------- | ---------------- |
| HMMs.jl    | LinearAlgebra[^2] | depends[^2]      | Threads[^1]      |
| HMMBase.jl | -                 | depends[^2]      | -                |
| hmmlearn   | NumPy[^2]         | NumPy[^2]        | NumPy[^2]        |
| hmmlearn   | PyTorch[^3]       | PyTorch[^3]      | PyTorch[^3]      |

[^1]: possibly affected by `JULIA_NUM_THREADS`
[^2]: possibly affected by `OPENBLAS_NUM_THREADS`
[^3]: possibly affected by `MKL_NUM_THREADS`

In addition, OpenBLAS threads have [negative interactions](https://github.com/JuliaLang/julia/issues/44201#issuecomment-1585656581) with Julia threads.
To overcome this obstacle, we run the Julia benchmarks (and only those) with `OPENBLAS_NUM_THREADS=1`.
