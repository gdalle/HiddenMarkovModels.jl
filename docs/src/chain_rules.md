# Chain rules

Suppose we are given observations $Y_1, ..., Y_T$, with hidden states $X_1, ..., X_T$.
Following [Rabiner1989](@cite), let $\pi \in \mathbb{R}^N$ denote the initial state distribution, $A \in \mathbb{R}^{N \times N}$ the transition matrix and $B \in \mathbb{R}^{N \times T}$ the matrix of statewise emission likelihoods.

## Vanilla forward-backward

The forward and backward variables are defined by

```math
\begin{align*}
\alpha_{i,t} & = \mathbb{P}(Y_{1:t}, X_t=i) \\
\beta_{i,t} & = \mathbb{P}(Y_{t+1:T} | X_t=i)
\end{align*}
```

They satisfy the following dynamic programming equations:

```math
\begin{align*}
\alpha_{j,t+1} & = \left(\sum_{i=1}^N \alpha_{i,t} a_{i,j}\right) b_{j,t+1} \\
\beta_{i,t} & = \sum_{j=1}^N a_{i,j} b_{j,t+1} \beta_{j,t+1}
\end{align*}
```

with initializations

```math
\begin{align*}
\alpha_{i,1} & = \pi_i b_{i,1} \\
\beta_{i,T} & = 1
\end{align*}
```

From them, we deduce the one-state and two-state marginals

```math
\begin{align*}
\gamma_{i,t} & = \mathbb{P}(X_t=i | Y_{1:T}) \propto \alpha_{i,t} \beta_{i,t} \\
\xi_{i,j,t} & = \mathbb{P}(X_t=i, X_{t+1}=j | Y_{1:T}) \propto \alpha_{i,t} a_{i,j} b_{j,t+1} \beta_{j,t+1}
\end{align*}
```

The likelihood of the whole sequence of observations is given by

```math
\mathcal{L} = \mathbb{P}(Y_{1:T}) = \sum_{i=1}^N \alpha_{i,T}
```

According to [Qin2000](@cite), derivatives can be obtained as follows:

```math
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \pi_i} &= \beta_{i,1} b_{i,1} \\
\frac{\partial \mathcal{L}}{\partial a_{i,j}} &= \sum_{t=1}^{T-1} \alpha_{i,t} b_{j,t+1} \beta_{j,t+1} \\
\frac{\partial \mathcal{L}}{\partial b_{j,t}} &= \begin{cases}
\pi_j \beta_{j,1} & \text{if $t = 1$} \\
\left(\sum_{i=1}^N \alpha_{i,t-1} a_{i,j}\right) \beta_{j,t} & \text{if $t > 1$}
\end{cases}
\end{align*}
```

## Scaled forward-backward

In this package, we use a slightly different version of the algorithm, including both the traditional scaling of [Rabiner1989](@cite) and a normalization of $B$ using $m_t = \max_i b_{i,t}$.

```math
\begin{align*}
\hat{\alpha}_{j,t+1} & = \left(\sum_{i=1}^N \bar{\alpha}_{i,t} a_{i,j}\right) \frac{b_{j,t+1}}{m_{t+1}} & c_t & = \frac{1}{\sum_j \hat{\alpha}_{j,t}} & \bar{\alpha}_{j,t} = c_t \hat{\alpha}_{j,t} \\
\hat{\beta}_{i,t} & = \sum_{j=1}^N a_{i,j} \frac{b_{j,t+1}}{m_{t+1}} \bar{\beta}_{j,t+1} & && \bar{\beta}_{j,t} = c_t \hat{\beta}_{j,t}
\end{align*}
```

with initializations

```math
\begin{align*}
\hat{\alpha}_{i,1} & = \pi_i \frac{b_{i,1}}{m_1} & c_1 & = \frac{1}{\sum_i \hat{\alpha}_{i,1}} & \bar{\alpha}_{i,1} & = c_1 \hat{\alpha}_{i,1} \\
\hat{\beta}_{i,T} & = 1 & && \bar{\beta}_{1,T} &= \hat{\beta}_{1,T} \text{ ??}
\end{align*}
```
