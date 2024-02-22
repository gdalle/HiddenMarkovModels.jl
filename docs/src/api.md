```@meta
CollapsedDocStrings = true
```

# API reference

```@docs
HiddenMarkovModels
```

## Sequence formatting

Most algorithms below ingest the data with two positional arguments `obs_seq` (mandatory) and `control_seq` (optional), and a keyword argument `seq_ends` (optional).

- If the data consists of a single sequence, `obs_seq` and `control_seq` are the corresponding vectors of observations and controls, and you don't need to provide `seq_ends`.
- If the data consists of multiple sequences, `obs_seq` and `control_seq` are concatenations of several vectors, whose end indices are given by `seq_ends`. Starting from separate sequences `obs_seqs` and `control_seqs`, you can run the following snippet:

```julia
obs_seq = reduce(vcat, obs_seqs)
control_seq = reduce(vcat, control_seqs)
seq_ends = cumsum(length.(obs_seqs))
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
joint_logdensityof
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

### Forward

```@docs
HiddenMarkovModels.ForwardStorage
HiddenMarkovModels.initialize_forward
HiddenMarkovModels.forward!
```

### Viterbi

```@docs
HiddenMarkovModels.ViterbiStorage
HiddenMarkovModels.initialize_viterbi
HiddenMarkovModels.viterbi!
```

### Forward-backward

```@docs
HiddenMarkovModels.ForwardBackwardStorage
HiddenMarkovModels.initialize_forward_backward
HiddenMarkovModels.forward_backward!
```

### Baum-Welch

```@docs
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
