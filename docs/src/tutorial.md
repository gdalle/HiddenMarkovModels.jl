# Tutorial

!!! warning "Work in progress"
    In the meantime, you can take a look at the files in `test`, which demonstrate more sophisticated ways to use the package.

```@repl tuto
using HiddenMarkovModels, Distributions
```

Constructing an HMM:

```@repl tuto
function random_gaussian_hmm(N)
    p = ones(N) / N  # initial distribution
    A = rand_trans_mat(N)  # transition matrix
    dists = [Normal(randn(), 1.0) for n in 1:N]  # observation distributions
    return HMM(p, A, dists)
end;
```

Checking its contents:

```@repl tuto
hmm = random_gaussian_hmm(3)
transition_matrix(hmm)
[obs_distribution(hmm, i) for i in 1:length(hmm)]
```

Simulating a sequence:

```@repl tuto
state_seq, obs_seq = rand(hmm, 1000);
first(state_seq, 10)'
first(obs_seq, 10)'
```

Computing the loglikelihood of an observation sequence:

```@repl tuto
logdensityof(hmm, obs_seq)
```

Inferring the most likely state sequence:

```@repl tuto
most_likely_state_seq = viterbi(hmm, obs_seq);
first(most_likely_state_seq, 10)'
```

Learning the parameters based on an observation sequence:

```@repl tuto
hmm_init = random_gaussian_hmm(3)
hmm_est, logL_evolution = baum_welch(hmm_init, obs_seq);
first(logL_evolution), last(logL_evolution)
transition_matrix(hmm_est)
[obs_distribution(hmm_est, i) for i in 1:length(hmm)]
```
