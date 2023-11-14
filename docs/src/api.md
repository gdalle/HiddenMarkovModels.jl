# API reference

```@docs
HiddenMarkovModels
```

## Types

```@docs
AbstractHiddenMarkovModel
HiddenMarkovModel
AbstractHMM
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
baum_welch
fit!
```

## Misc

```@docs
rand_prob_vec
rand_trans_mat
HiddenMarkovModels.fit_element_from_sequence!
HiddenMarkovModels.LightDiagNormal
```

## In-place algorithms (internals)

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
HiddenMarkovModels.initialize_logL_evolution
```

### Modifying storage

```@docs
HiddenMarkovModels.forward!
HiddenMarkovModels.viterbi!
HiddenMarkovModels.forward_backward!
HiddenMarkovModels.baum_welch!
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
- `d` or `dists`: observation distribution (vector of `rand`-able and `logdensityof`-able objects)
- `state_seq`: a sequence of states (vector of integers)
- `obs_seq`: a sequence of observations (vector of individual observations)
- `obs_seqs`: several sequences of observations
- `nb_seqs`: number of observation sequences

### Forward backward

- `(log)b`: vector of observation (log)likelihoods by state for an individual observation
- `(log)B`: matrix of observation (log)likelihoods by state for a sequence of observations
- `α`: scaled forward variables
- `β`: scaled backward variables
- `γ`: state marginals
- `ξ`: transition marginals
- `logL`: loglikelihood of a sequence of observations

## Index

```@index
```
