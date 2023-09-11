# Tutorial

```@repl tuto
using HiddenMarkovModels
using Distributions
```

## Using the built-in HMM

Constructing a model:

```@repl tuto
function random_gaussian_hmm(N)
    p = ones(N) / N  # initial distribution
    A = rand_trans_mat(N)  # transition matrix
    dists = [Normal(randn(), 1.0) for n in 1:N]  # observation distributions
    return HMM(p, A, dists)
end;
```

Checking its contents:

```@repl tuto
hmm = random_gaussian_hmm(3)
transition_matrix(hmm)
[obs_distribution(hmm, i) for i in 1:length(hmm)]
```

Simulating a sequence:

```@repl tuto
state_seq, obs_seq = rand(hmm, 1000);
first(state_seq, 10)'
first(obs_seq, 10)'
```

Computing the loglikelihood of an observation sequence:

```@repl tuto
logdensityof(hmm, obs_seq)
```

Inferring the most likely state sequence:

```@repl tuto
most_likely_state_seq = viterbi(hmm, obs_seq);
first(most_likely_state_seq, 10)'
```

Learning the parameters based on an observation sequence:

```@repl tuto
hmm_init = random_gaussian_hmm(3)
hmm_est, logL_evolution = baum_welch(hmm_init, obs_seq);
first(logL_evolution), last(logL_evolution)
transition_matrix(hmm_est)
[obs_distribution(hmm_est, i) for i in 1:length(hmm)]
```

## Making your own HMM

The built-in HMM is perfect when the initial state distribution, transition matrix and emission distributions are separate objects, which means their re-estimation can be done separately.
But in some cases these parameters might be correlated.
For instance, you may want an HMM whose initial state distribution always corresponds to the equilibrium distribution associated with the transition matrix.

In such cases, it is necessary to implement a new subtype of [`AbstractHMM`](@ref) with all its required methods.
To ascertain that a type indeed satisfies the interface, you can use [RequiredInterfaces.jl](https://github.com/Seelengrab/RequiredInterfaces.jl) as follows:

```@repl tuto
using RequiredInterfaces: check_interface_implemented
check_interface_implemented(AbstractHMM, HMM)
```

And of course, if your implementation is insufficient, the test will fail:

```@repl tuto
struct EmptyHMM end
check_interface_implemented(AbstractHMM, EmptyHMM)
```

Note that this test does not check the `fit!` method.
Since it is only used in the Baum-Welch algorithm, it is an optional part of the `AbstractHMM` interface.
