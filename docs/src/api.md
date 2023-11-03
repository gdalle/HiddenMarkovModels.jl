# API reference

```@docs
HiddenMarkovModels
```

## Types

```@docs
AbstractHiddenMarkovModel
HiddenMarkovModel
AbstractHMM
PermutedHMM
HMM
```

## Basics

```@docs
rand
length
eltype
initialization
transition_matrix
obs_distributions
```

## Inference

```@docs
logdensityof
forward
viterbi
forward_backward
```

## Learning

```@docs
fit!
fit
baum_welch
```

## Internals

```@docs
HiddenMarkovModels.ForwardBackwardStorage
HiddenMarkovModels.fit_element_from_sequence!
HiddenMarkovModels.LightDiagNormal
```

## Notations

### Integers

- `N`: number of states
- `D`: dimension of the observations
- `T`: trajectory length
- `K`: number of trajectories

### Models and simulations

- `p` or `init`: initialization (vector of state probabilities)
- `A` or `trans`: transition_matrix (matrix of transition probabilities)
- `dists`: observation distribution (vector of `rand`-able and `logdensityof`-able objects)
- `state_seq`: a sequence of states (vector of integers)
- `obs_seq`: a sequence of observations (vector of individual observations)
- `obs_seqs`: several sequences of observations

### Forward backward

- `(log)b`: vector of observation (log)likelihoods by state for an individual observation
- `(log)B`: matrix of observation (log)likelihoods by state for a sequence of observations
- `α`: scaled forward variables
- `β`: scaled backward variables
- `γ`: one-state marginals
- `ξ`: two-state marginals
- `logL`: loglikelihood of a sequence of observations

## Index

```@index
```
