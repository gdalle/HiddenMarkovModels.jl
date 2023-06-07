# Notations

Our whole package is based on the following paper by Rabiner (1989):

> [A tutorial on hidden Markov models and selected applications in speech recognition](https://ieeexplore.ieee.org/document/18626)

Please refer to it for mathematical explanations.

## State process

- `sp` or `state_process`: a `StateProcess`
- `p`: initial_distribution (vector of state probabilities)
- `A`: transition_matrix (matrix of transition probabilities)
- `state_seq`: a sequence of states (vector of integers)

## Observation process

- `op` or `obs_process`: an `ObservationProcess`
- `(log)b`: vector of observation (log)likelihoods by state for a single observation
- `(log)B`: matrix of observation (log)likelihoods by state for a sequence of observations
- `obs_seq`: a sequence of observations
- `obs_seqs`: several sequences of observations

## Forward backward

- `α`: forward variables
- `c`: forward variable inverse normalizations
- `β`: backward variables
- `γ`: one-state marginals
- `ξ`: two-state marginals
- `logL`: loglikelihood of a sequence of observations
