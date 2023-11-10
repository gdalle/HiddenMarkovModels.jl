# Alternatives - performance

We compare performance among the following packages:

- HiddenMarkovModels.jl (abbreviated to HMMs.jl)
- [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl)
- [hmmlearn](https://github.com/hmmlearn/hmmlearn)
- [pomegranate](https://github.com/jmschrei/pomegranate)

The test case is an HMM with diagonal multivariate normal observations.

- `N`: number of states
- `D`: dimension of the observations
- `T`: trajectory length
- `K`: number of trajectories
- `I`: number of Baum-Welch iterations

## Reproducibility

These benchmarks were generated in the following environment: [`setup.txt`](./assets/benchmark/results/setup.txt).

If you want to run them on your machine:

1. Clone the [HiddenMarkovModels.jl](https://github.com/gdalle/HiddenMarkovModels.jl) repository
2. Open a Julia REPL at the root
3. Run the following commands

   ```julia
   include("benchmark/run_benchmarks.jl")
   include("benchmark/process_benchmarks.jl")
   ```

## Remarks

### Julia-to-Python overhead

Since the Python packages are called from Julia with [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl), we pay a small overhead that is hard to quantify.
On the plots, we compensate it by subtracting the runtime of the same algorithm for the smallest instance (`N=1`, `D=1`, `T=2`, `K=1`, `I=1`) from all Python-generated curves.

### Allocations

A major bottleneck of performance in Julia is memory allocations. The benchmarks for HMMs.jl thus employ a custom implementation of diagonal multivariate normals, which is entirely allocation-free.

This partly explains the performance gap with HMMBase.jl as the dimension `D` grows beyond 1.
Such a trick is also possible with HMMBase.jl, but more demanding since it requires subtyping `Distribution` from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl), instead of just implementing [DensityInterface.jl](https://github.com/JuliaMath/DensityInterface.jl).

### Parallelism

The packages we include have different approaches to parallelism, which can bias the evaluation in complex ways:

| Package     | States `N`        | Observations `D` | Sequences `K` |
| ----------- | ----------------- | ---------------- | ------------- |
| HMMs.jl     | LinearAlgebra[^2] | depends[^2]      | Threads[^1]   |
| HMMBase.jl  | -                 | depends[^2]      | -             |
| hmmlearn    | NumPy[^2]         | NumPy[^2]        | NumPy[^2]     |
| pomegranate | PyTorch[^3]       | PyTorch[^3]      | PyTorch[^3]   |

[^1]: affected by number of Julia threads
[^2]: affected by number of OpenBLAS threads
[^3]: affected by number of Pytorch threads

We report each number of threads in [`setup.txt`](./assets/benchmark/results/setup.txt).
Since OpenBLAS threads have [negative interactions](https://github.com/JuliaLang/julia/pull/50124) with Julia threads, we run the Julia benchmarks with a single OpenBLAS thread.

## Numerical results

### Low dimension

Full benchmark logs: [`low_dim.csv`](./assets/benchmark/results/low_dim.csv).

![](./assets/benchmark/plots/low_dim_logdensity_(D=1,T=1000,K=1).svg)
![](./assets/benchmark/plots/low_dim_viterbi_(D=1,T=1000,K=1).svg)
![](./assets/benchmark/plots/low_dim_forward_backward_(D=1,T=1000,K=1).svg)
![](./assets/benchmark/plots/low_dim_baum_welch_(D=1,T=1000,K=1,I=10).svg)

_Here, pomegranate is not included because it is much slower on very small inputs._

### High dimension

Full benchmark logs: [`high_dim.csv`](./assets/benchmark/results/high_dim.csv).

![](./assets/benchmark/plots/high_dim_logdensity_(D=10,T=200,K=50).svg)
![](./assets/benchmark/plots/high_dim_viterbi_(D=10,T=200,K=50).svg)
![](./assets/benchmark/plots/high_dim_forward_backward_(D=10,T=200,K=50).svg)
![](./assets/benchmark/plots/high_dim_baum_welch_(D=10,T=200,K=50,I=10).svg)

_Here, HMMBase.jl is not included because it does not support multiple sequences._
