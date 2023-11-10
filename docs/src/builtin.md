# Tutorial - built-in HMM

```@example tuto
using Distributions
using HiddenMarkovModels

using Random; Random.seed!(63)
```

## Construction

Creating a model:

```@example tuto
function gaussian_hmm(N; noise=0)
    p = ones(N) / N  # initial distribution
    A = rand_trans_mat(N)  # transition matrix
    d = [Normal(i + noise * randn(), 0.5) for i in 1:N]  # observation distributions
    return HMM(p, A, d)
end
```

Checking its contents:

```@example tuto
N = 3
hmm = gaussian_hmm(N)
transition_matrix(hmm)
```

```@example tuto
hmm.dists
```

Simulating a sequence:

```@example tuto
T = 1000
state_seq, obs_seq = rand(hmm, T);
first(state_seq, 10)'
```

```@example tuto
first(obs_seq, 10)'
```

## Inference

Computing the loglikelihood of an observation sequence:

```@example tuto
logdensityof(hmm, obs_seq)
```

Inferring the most likely state sequence:

```@example tuto
most_likely_state_seq = viterbi(hmm, obs_seq);
first(most_likely_state_seq, 10)'
```

Learning the parameters based on an observation sequence:

```@example tuto
hmm_init = gaussian_hmm(N, noise=1)
hmm_est, logL_evolution = baum_welch(hmm_init, obs_seq);
first(logL_evolution), last(logL_evolution)
```

Correcting state order because we know observation means are increasing in the true model:

```@example tuto
d_est = hmm_est.dists
```

```@example tuto
perm = sortperm(1:3, by=i->d_est[i].μ)
```

```@example tuto
hmm_est = HiddenMarkovModels.PermutedHMM(hmm_est, perm)
```

Evaluating errors:

```@example tuto
cat(transition_matrix(hmm_est), transition_matrix(hmm), dims=3)
```
