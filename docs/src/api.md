# API reference

```@docs
HiddenMarkovModels
```

## Types

```@docs
AbstractHMM
HMM
```

## Interface

```@docs
initialization
transition_matrix
obs_distributions
```

## Utils

```@docs
length
rand
eltype
seq_limits
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
baum_welch
fit!
```

## In-place versions

```@docs
HiddenMarkovModels.ForwardStorage
HiddenMarkovModels.initialize_forward
HiddenMarkovModels.forward!
HiddenMarkovModels.ViterbiStorage
HiddenMarkovModels.initialize_viterbi
HiddenMarkovModels.viterbi!
HiddenMarkovModels.ForwardBackwardStorage
HiddenMarkovModels.initialize_forward_backward
HiddenMarkovModels.forward_backward!
HiddenMarkovModels.baum_welch!
```

## Misc

```@docs
HiddenMarkovModels.rand_prob_vec
HiddenMarkovModels.rand_trans_mat
HiddenMarkovModels.LightDiagNormal
HiddenMarkovModels.LightCategorical
HiddenMarkovModels.fit_in_sequence!
```

## Index

```@index
```
