# API reference

```@docs
HiddenMarkovModels
HMMs
```

## Types

```@docs
HiddenMarkovModel
HMM
HMMs.StateProcess
HMMs.StandardStateProcess
HMMs.ObservationProcess
HMMs.StandardObservationProcess
```

## Basics

```@docs
length
rand
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
baum_welch
```

## Reimplement if needed

```@docs
HMMs.fit!
HMMs.fit_element_from_sequence!
HMMs.LightDiagNormal
```

## Index

```@index
```

## Notations

### Integers

- `N`: number of states
- `D`: dimension of the observations
- `T`: trajectory length
- `K`: number of trajectories

### State process

- `sp` or `state_process`: a `StateProcess`
- `p`: initial_distribution (vector of state probabilities)
- `A`: transition_matrix (matrix of transition probabilities)
- `state_seq`: a sequence of states (vector of integers)

### Observation process

- `op` or `obs_process`: an `ObservationProcess`
- `(log)b`: vector of observation (log)likelihoods by state for an individual observation
- `(log)B`: matrix of observation (log)likelihoods by state for a sequence of observations
- `obs_seq`: a sequence of observations (vector of individual observations)
- `obs_seqs`: several sequences of observations

### Forward backward

- `α`: forward variables
- `c`: forward variable inverse normalizations
- `β`: backward variables
- `γ`: one-state marginals
- `ξ`: two-state marginals
- `logL`: loglikelihood of a sequence of observations