# API reference

```@docs
HiddenMarkovModels
HMMs
```

## Types

```@docs
HMMs.AbstractModel
```

### Markov chains

```@docs
AbstractMarkovChain
MarkovChain
AbstractMC
MC
```

### Hidden Markov Models

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
initial_distribution
transition_matrix
obs_distribution
```

## Inference

```@docs
logdensityof
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
HMMs.ForwardBackwardStorage
HMMs.fit_element_from_sequence!
HMMs.LightDiagNormal
```

## Notations

### Integers

- `N`: number of states
- `D`: dimension of the observations
- `T`: trajectory length
- `K`: number of trajectories

### Models and simulations

- `p` or `init`: initial_distribution (vector of state probabilities)
- `A` or `trans`: transition_matrix (matrix of transition probabilities)
- `dists`: observation distribution (vector of `rand`-able and `logdensityof`-able objects)
- `state_seq`: a sequence of states (vector of integers)
- `obs_seq`: a sequence of observations (vector of individual observations)
- `obs_seqs`: several sequences of observations

### Forward backward

- `(log)b`: vector of observation (log)likelihoods by state for an individual observation
- `(log)B`: matrix of observation (log)likelihoods by state for a sequence of observations
- `α`: forward variables
- `β`: backward variables
- `γ`: one-state marginals
- `ξ`: two-state marginals
- `logL`: loglikelihood of a sequence of observations

## Index

```@index
```