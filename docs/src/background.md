# Background

## What are HMMs?

[Hidden Markov Models](https://en.wikipedia.org/wiki/Hidden_Markov_model) (HMMs for short) are a statistical modeling framework that is ubiquitous in signal processing, bioinformatics and plenty of other fields. They capture the distribution of an observation sequence $(Y_t)$ by assuming the existence of a latent state sequence $(X_t)$ such that:

* the sequence $(X_t)$ follows a (discrete time, discrete space) Markov chain
* for each $t$, the distribution of $Y_t$ is entirely determined by the value of $X_t$

## What can we do with them?

Imagine we are given an observation sequence $(Y_t)$ and a parametric family of HMMs $\{\mathbb{P}_\theta : \theta \in \Theta\}$.
We can list three fundamental problems, each of which has a solution that relies on dynamic programming:

| Problem    | Goal                                                                                                      | Solution                  |
| ---------- | --------------------------------------------------------------------------------------------------------- | ------------------------- |
| Evaluation | Likelihood of the observation sequence $\mathbb{P}_\theta(Y_{1:T})$                                       | Forward algorithm         |
| Decoding   | Most likely state sequence $\underset{X_{1:T}}{\mathrm{argmax}}~\mathbb{P}_\theta(X_{1:T} \vert Y_{1:T})$ | Viterbi algorithm         |
| Learning   | Best parameter $\underset{\theta}{\mathrm{argmax}}~\mathbb{P}_\theta(Y_{1:T})$                                               | Baum-Welch algorithm |

Our whole package is based on the tutorial by [Rabiner1989](@cite), you can refer to it for more details.

## Bibliography

```@bibliography
```
