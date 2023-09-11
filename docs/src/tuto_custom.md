# Tutorial - custom HMM

```@repl tuto
using HiddenMarkovModels
using Distributions
```

The built-in HMM is perfect when the initial state distribution, transition matrix and emission distributions are separate objects, which means their re-estimation can be done separately.
But in some cases these parameters might be correlated.
For instance, you may want an HMM whose initial state distribution always corresponds to the equilibrium distribution associated with the transition matrix.

## Interface

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

## Example (coming soon)