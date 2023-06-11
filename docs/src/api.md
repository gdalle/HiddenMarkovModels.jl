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
