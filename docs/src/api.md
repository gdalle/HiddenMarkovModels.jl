# API reference

```@docs
HiddenMarkovModels
```

## Types

```@docs
AbstractHMM
HMM
PermutedHMM
```

## Basics

```@docs
rand
length
eltype
initialization
transition_matrix
obs_distributions
fit!
```

## Inference

```@docs
logdensityof
forward
viterbi
forward_backward
baum_welch
MultiSeq
```

## Internals

These objects are not yet stabilized and may change at any time.
Do not consider them to be part of the API subject to semantic versioning.

### Storage types

```@docs
HiddenMarkovModels.ForwardStorage
HiddenMarkovModels.ViterbiStorage
HiddenMarkovModels.ForwardBackwardStorage
HiddenMarkovModels.BaumWelchStorage
```

### Initializing storage

```@docs
HiddenMarkovModels.initialize_forward
HiddenMarkovModels.initialize_viterbi
HiddenMarkovModels.initialize_forward_backward
HiddenMarkovModels.initialize_baum_welch
```

### Modifying storage

```@docs
HiddenMarkovModels.forward!
HiddenMarkovModels.viterbi!
HiddenMarkovModels.forward_backward!
HiddenMarkovModels.baum_welch!
```

## Misc

```@docs
HiddenMarkovModels.rand_prob_vec
HiddenMarkovModels.rand_trans_mat
HiddenMarkovModels.project_prob_vec
HiddenMarkovModels.project_trans_mat
HiddenMarkovModels.fit_in_sequence!
HiddenMarkovModels.LightDiagNormal
HiddenMarkovModels.LightCategorical
```

## Notations

### Integers

- `N`: number of states
- `T`: trajectory length

### Models and simulations

- `init`: initialization (vector of state probabilities)
- `trans`: transition_matrix (matrix of transition probabilities)
- `dists`: observation distribution (vector of `rand`-able and `logdensityof`-able objects)
- `state_seq`: a sequence of states (vector of integers)
- `obs_seq`: a sequence of observations (vector of individual observations)
- `obs_seqs`: several sequences of observations
- `nb_seqs`: number of observation sequences
- `logL`: loglikelihood

### Forward backward

- `(log)b`: vector of observation (log)likelihoods by state for an individual observation
- `(log)B`: matrix of observation (log)likelihoods by state for a sequence of observations
- `α`: scaled forward variables
- `β`: scaled backward variables
- `γ`: state marginals
- `ξ`: transition marginals

## Index

```@index
```
