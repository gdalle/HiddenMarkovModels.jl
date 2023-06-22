# Background

## What is an HMM?

[Hidden Markov Models](https://en.wikipedia.org/wiki/Hidden_Markov_model) (HMMs for short) are a statistical modeling framework that is ubiquitous in signal processing, bioinformatics and plenty of other fields. They capture the distribution of an observation sequence $(Y_t)$ by assuming the existence of a latent state sequence $(X_t)$ such that:
* the sequence $(X_t)$ follows a (discrete time, discrete space) Markov chain
* for each $t$, the distribution of $Y_t$ is entirely determined by the value of $X_t$

## What can we do with it?

Imagine we are given an observation sequence $(Y_t)$ and a parametric family of HMMs $\{p_\theta \colon \theta \in \Theta\}$.
We can list three fundamental problems, each of which has a solution that relies on dynamic programming:

| Problem    | Goal                                                                             | Solution                  |
| ---------- | -------------------------------------------------------------------------------- | ------------------------- |
| Evaluation | Likelihood of the observation sequence $p_\theta(Y)$ for a fixed $\theta$        | Forward algorithm         |
| Decoding   | Most likely state sequence $\arg\max_X p_\theta(X \vert Y)$ for a fixed $\theta$ | Viterbi algorithm         |
| Learning   | Best parameter $\arg\max_\theta p_\theta(Y)$                                     | Baum-Welch (EM) algorithm |

## Going further

Our whole package is based on the following paper by Rabiner (1989):

> [A tutorial on hidden Markov models and selected applications in speech recognition](https://ieeexplore.ieee.org/document/18626)

Please refer to it for more details.
