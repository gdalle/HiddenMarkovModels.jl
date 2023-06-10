# Benchmarks


The test case is an HMM with multi-dimensional Gaussian observations, initialized randomly.
We compare the following packages:

- HiddenMarkovModels.jl (abbreviated to HMMs.jl)
- [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl)
- [hmmlearn](https://github.com/hmmlearn/hmmlearn)
- [pomegranate](https://github.com/jmschrei/pomegranate)

For now, pomegranate is not included on the plots because it is much slower on very small inputs.

## Notations

- ``N``: number of states
- ``D``: dimension of the Gaussian observations
- ``T``: trajectory length
- ``K``: number of trajectories
- ``I``: iterations in the Baum-Welch algorithm

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

## Remarks on parallelism

The packages we include have different approaches to parallelism, which can bias the evaluation in complex ways:

| Package    | States $N$        | Observations $D$ | Trajectories $K$ |
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