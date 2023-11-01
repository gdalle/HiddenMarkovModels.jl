# Tutorial - custom HMM

The built-in HMM is perfect when the initial state distribution, transition matrix and emission distributions can be updated independently with Maximum Likelihood Estimation.
But some of these parameters might be correlated, or fixed.
Or they might come with a prior, which forces you to use Maximum A Posteriori instead.

In such cases, it is necessary to implement a new subtype of [`AbstractHMM`](@ref) with all its required methods.

```@example tuto
using Distributions
using HiddenMarkovModels
using LinearAlgebra
using RequiredInterfaces
using StatsAPI

using Random; Random.seed!(63)
```

## Interface

To ascertain that a type indeed satisfies the interface, you can use [RequiredInterfaces.jl](https://github.com/Seelengrab/RequiredInterfaces.jl).

```@example tuto
RequiredInterfaces.check_interface_implemented(AbstractHMM, HMM)
```

If your implementation is insufficient, the test will list missing methods.

```@example tuto
struct EmptyHMM end
RequiredInterfaces.check_interface_implemented(AbstractHMM, EmptyHMM)
```

Note that this test does not check the `StatsAPI.fit!` method.
Since it is only used in the Baum-Welch algorithm, it is an optional part of the `AbstractHMM` interface.

## Example

We show how to implement an HMM whose initial distribution is always the equilibrium distribution of the underlying Markov chain.
The code that follows is not efficient (it leads to a lot of allocations), but it would be fairly easy to optimize if needed.

The equilibrium distribution of a Markov chain is the (only) left eigenvector associated with the left eigenvalue $1$.

```@example tuto
function markov_equilibrium(A)
    p = real.(eigvecs(A')[:, end])
    return p ./ sum(p)
end
```

We now define our custom HMM by taking inspiration from `src/types/hmm.jl` and making a few modifications:

```@example tuto
struct EquilibriumHMM{R,D} <: AbstractHMM
    trans::Matrix{R}
    dists::Vector{D}
end
```

The interface is only different as far as the initialization is concerned.

```@example tuto
Base.length(hmm::EquilibriumHMM) = length(hmm.dists)
HMMs.initialization(hmm::EquilibriumHMM) = markov_equilibrium(hmm.trans)  # this is new
HMMs.transition_matrix(hmm::EquilibriumHMM) = hmm.trans
HMMs.obs_distribution(hmm::EquilibriumHMM, i::Integer) = hmm.dists[i]
```

As for fitting, we simply ignore the initialization count and copy the rest of the original code (with a few simplifications):

```@example tuto
function StatsAPI.fit!(
    hmm::EquilibriumHMM{R,D}, obs_seqs, fbs
) where {R,D}
    hmm.trans .= trans_count ./ sum(trans_count, dims=2)
    obs_seqs_concat = reduce(vcat, obs_seqs)
    state_marginals_concat = reduce(hcat, fb.γ for fb in fbs)
    for i in 1:N
        hmm.dists[i] = fit(D, obs_seqs_concat, state_marginals_concat[i, :])
    end
end
```

Let's take our new model for a spin:

```@example tuto
function gaussian_equilibrium_hmm(N; noise=0)
    A = rand_trans_mat(N)
    dists = [Normal(i + noise * randn(), 0.5) for i in 1:N]
    return EquilibriumHMM(A, dists)
end;
```

```@example tuto
N = 3
hmm = gaussian_equilibrium_hmm(N);
transition_matrix(hmm)
```

```@example tuto
[obs_distribution(hmm, i) for i in 1:N]
```

We can estimate parameters based on several observation sequences.
Note that as soon as we tamper with the re-estimation procedure, the loglikelihood is no longer guaranteed to increase during Baum-Welch, which is why we turn off the corresponding check.

```@example tuto
T = 1000
nb_seqs = 10
obs_seqs = [rand(hmm, T).obs_seq for _ in 1:nb_seqs]

hmm_init = gaussian_equilibrium_hmm(N; noise=1)
hmm_est, logL_evolution = baum_welch(
    hmm_init, obs_seqs, nb_seqs; check_loglikelihood_increasing=false
);
first(logL_evolution), last(logL_evolution)
```

Let's correct the state order:

```@example tuto
[obs_distribution(hmm_est, i) for i in 1:N]
```

```@example tuto
perm = sortperm(1:3, by=i->obs_distribution(hmm_est, i).μ)
```

```@example tuto
hmm_est = PermutedHMM(hmm_est, perm)
```

And finally evaluate the errors:

```@example tuto
cat(transition_matrix(hmm_est), transition_matrix(hmm), dims=3)
```