# Tutorial

!!! warning "Work in progress"
    In the meantime, you can take a look at the files in `test` (especially `test/correctness.jl`) which demonstrate various ways in which the package can be used.

```@repl
using HiddenMarkovModels, Distributions

function random_gaussian_hmm(N)
    p, A = ones(N) / N, rand_trans_mat(N)
    μ, σ = randn(N), ones(N)
    dists = [Normal(μ[n], σ[n]) for n in 1:N]
    return HMM(p, A, dists)
end;

hmm = random_gaussian_hmm(2)  # initialization

state_seq, obs_seq = rand(hmm, 1000);  # simulation

logdensityof(hmm, obs_seq)  # loglikelihood

most_likely_state_seq = viterbi(hmm, obs_seq);  # inference

hmm_est, logL_evolution = baum_welch(random_gaussian_hmm(2), [obs_seq]);  # estimation
hmm_est
first(logL_evolution), last(logL_evolution)
```