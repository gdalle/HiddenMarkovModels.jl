# Forward-backward

Suppose we are given observations $Y_1, ..., Y_T$, with hidden states $X_1, ..., X_T$.
Following [Rabiner1989](@cite), let $\pi \in \mathbb{R}^N$ denote the initial state distribution, $A \in \mathbb{R}^{N \times N}$ the transition matrix and $B \in \mathbb{R}^{N \times T}$ the matrix of statewise observation likelihoods.

## Vanilla forward-backward

### Recursion

The forward and backward variables are defined by

```math
\begin{align*}
\alpha_{i,t} & = \mathbb{P}(Y_{1:t}, X_t=i) \\
\beta_{i,t} & = \mathbb{P}(Y_{t+1:T} | X_t=i)
\end{align*}
```

They are initialized with

```math
\begin{align*}
\alpha_{i,1} & = \pi_i b_{i,1} \\
\beta_{i,T} & = 1
\end{align*}
```

and satisfy the dynamic programming equations

```math
\begin{align*}
\alpha_{j,t+1} & = \left(\sum_{i=1}^N \alpha_{i,t} a_{i,j}\right) b_{j,t+1} \\
\beta_{i,t} & = \sum_{j=1}^N a_{i,j} b_{j,t+1} \beta_{j,t+1}
\end{align*}
```

### Likelihood

The likelihood of the whole sequence of observations is given by

```math
\mathcal{L} = \mathbb{P}(Y_{1:T}) = \sum_{i=1}^N \alpha_{i,T}
```

### Marginals

From the forward and backward variables, we deduce the one-state and two-state marginals

```math
\begin{align*}
\gamma_{i,t} & = \mathbb{P}(X_t=i | Y_{1:T}) = \frac{\alpha_{i,t} \beta_{i,t}}{\sum_i \alpha_{i,t} \beta_{i,t}} \\
\xi_{i,j,t} & = \mathbb{P}(X_t=i, X_{t+1}=j | Y_{1:T}) = \frac{\alpha_{i,t} a_{i,j} b_{j,t+1} \beta_{j,t+1}}{\sum_{i,j} \alpha_{i,t} a_{i,j} b_{j,t+1} \beta_{j,t+1}}
\end{align*}
```

The denominator in both cases is equal to the likelihood $\mathcal{L}$.

### Derivatives

According to [Qin2000](@cite), derivatives can be obtained as follows:

```math
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \pi_i} &= \beta_{i,1} b_{i,1} \\
\frac{\partial \mathcal{L}}{\partial a_{i,j}} &= \sum_{t=1}^{T-1} \alpha_{i,t} b_{j,t+1} \beta_{j,t+1} \\
\frac{\partial \mathcal{L}}{\partial b_{j,1}} &= \pi_j \beta_{j,1} \\
\frac{\partial \mathcal{L}}{\partial b_{j,t}} &= \left(\sum_{i=1}^N \alpha_{i,t-1} a_{i,j}\right) \beta_{j,t} 
\end{align*}
```

## Scaled forward-backward

In this package, we use a slightly different version of the algorithm, including both the traditional scaling of [Rabiner1989](@cite) and a normalization of $B$ using $m_t = \max_i b_{i,t}$.

### Recursion

The variables are initialized with

```math
\begin{align*}
\hat{\alpha}_{i,1} & = \pi_i \frac{b_{i,1}}{m_1} & c_1 & = \frac{1}{\sum_i \hat{\alpha}_{i,1}} & \bar{\alpha}_{i,1} & = c_1 \hat{\alpha}_{i,1} \\
\hat{\beta}_{i,T} & = 1 & && \bar{\beta}_{1,T} &= c_T \hat{\beta}_{1,T}
\end{align*}
```

and satisfy the dynamic programming equations

```math
\begin{align*}
\hat{\alpha}_{j,t+1} & = \left(\sum_{i=1}^N \bar{\alpha}_{i,t} a_{i,j}\right) \frac{b_{j,t+1}}{m_{t+1}} & c_{t+1} & = \frac{1}{\sum_j \hat{\alpha}_{j,t+1}} & \bar{\alpha}_{j,t+1} = c_{t+1} \hat{\alpha}_{j,t+1} \\
\hat{\beta}_{i,t} & = \sum_{j=1}^N a_{i,j} \frac{b_{j,t+1}}{m_{t+1}} \bar{\beta}_{j,t+1} & && \bar{\beta}_{j,t} = c_t \hat{\beta}_{j,t}
\end{align*}
```

In terms of the original variables, we find

```math
\begin{align*}
\bar{\alpha}_{i,t} &= \alpha_{i,t} \prod_{s=1}^t \frac{c_s}{m_s} \\
\bar{\beta}_{i,t} &= \beta_{i,t} c_t \prod_{s=t+1}^T \frac{c_s}{m_s}
\end{align*}
```

### Likelihood

However, the formula for the likelihood differs:

```math
1 = \sum_{i=1}^N \bar{\alpha}_{i,T} = \left(\prod_{t=1}^T \frac{c_t}{m_t}\right) \sum_{i=1}^N \alpha_{i,T}
```

which means

```math
\begin{align*}
\mathcal{L} & = \sum_{i=1}^N \alpha_{i,T} = \prod_{t=1}^T \frac{m_t}{c_t} \\
\log \mathcal{L} & = \sum_{t=1}^T \log m_t - \sum_{t=1}^T \log c_t \\
\end{align*}
```

### Marginals

We then compute the marginals using adjusted formulas

```math
\begin{align*}
\bar{\gamma}_{i,t} & \propto \bar{\alpha}_{i,t} \bar{\beta}_{i,t} \\
\bar{\xi}_{i,j,t} & \propto \bar{\alpha}_{i,t} a_{i,j} \frac{b_{j,t+1}}{m_{t+1}} \bar{\beta}_{j,t+1}
\end{align*}
```

The scaled variables differ by a multiplicative factor that does not depend on $i$, which means that we recover the exact same marginals after normalization: $\bar{\gamma} = \gamma$ and $\bar{\xi} = \xi$.

We can even compute the normalizations explicitly:

```math
\begin{align*}
\sum_{i=1}^N \bar{\gamma}_{i,t} &= \sum_{i=1}^N \left(\alpha_{i,t} \prod_{s=1}^t \frac{c_s}{m_s}\right) \left(\beta_{i,t} c_t \prod_{s=t+1}^T \frac{c_s}{m_s}\right) \\
&= \left(\sum_{i=1}^N \alpha_{i,t} \beta_{i,t}\right) c_t \prod_{s=1}^T \frac{c_s}{m_s} = \mathcal{L} c_t \frac{1}{\mathcal{L}} = c_t
\end{align*}
```

```math
\begin{align*}
\sum_{i,j=1}^N \bar{\xi}_{i,j,t} &= \sum_{i,j=1}^N \left(\alpha_{i,t} \prod_{s=1}^t \frac{c_s}{m_s}\right) a_{i,j} \frac{b_{j,t+1}}{m_{t+1}} \left(\beta_{j,t+1} c_{t+1} \prod_{s=t+2}^T \frac{c_s}{m_s}\right) \\
&= \left(\sum_{i,j=1}^N \alpha_{i,t} a_{i,j} b_{j,t+1} \beta_{j,t+1}\right) \prod_{s=1}^T \frac{c_s}{m_s} = \mathcal{L} \frac{1}{\mathcal{L}} = 1
\end{align*}
```

### Derivatives

And we also need to adapt the derivatives.
For the initial distribution,

```math
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \pi_i} &= \beta_{i,1} b_{i,1} = \left(\bar{\beta}_{i,1} \frac{1}{c_1} \prod_{s=2}^T \frac{m_s}{c_s} \right) b_{i,1} \\
&= \left(\prod_{s=1}^T \frac{m_s}{c_s}\right) \bar{\beta}_{i,1} \frac{b_{i,1}}{m_1}  = \mathcal{L} \bar{\beta}_{i,1} \frac{b_{i,1}}{m_1} 
\end{align*}
```

For the transition matrix,

```math
\begin{align*}
\frac{\partial \mathcal{L}}{\partial a_{i,j}} &= \sum_{t=1}^{T-1} \alpha_{i,t} b_{j,t+1} \beta_{j,t+1} \\
&= \sum_{t=1}^{T-1} \left(\bar{\alpha}_{i,t} \prod_{s=1}^t \frac{m_s}{c_s} \right) b_{j,t+1} \left(\bar{\beta}_{j,t+1} \frac{1}{c_{t+1}} \prod_{s=t+2}^T \frac{m_s}{c_s} \right) \\
&= \sum_{t=1}^{T-1} \left(\prod_{s=1}^T \frac{m_s}{c_s} \right) \bar{\alpha}_{i,t} \frac{b_{j,t+1}}{m_{t+1}} \bar{\beta}_{j,t+1} \\
&= \mathcal{L} \sum_{t=1}^{T-1} \bar{\alpha}_{i,t} \frac{b_{j,t+1}}{m_{t+1}} \bar{\beta}_{j,t+1} \\
\end{align*}
```

And for the statewise observation likelihoods,

```math
\begin{align*}
\frac{\partial \mathcal{L}}{\partial b_{j,1}} &= \pi_j \beta_{j,1} = \pi_j \bar{\beta}_{j,1} \frac{1}{c_1} \prod_{s=2}^T \frac{m_s}{c_s} = \mathcal{L} \pi_j \bar{\beta}_{j,1} \frac{1}{m_1}
\end{align*}
```

```math
\begin{align*}
\frac{\partial \mathcal{L}}{\partial b_{j,t}} &= \left(\sum_{i=1}^N \alpha_{i,t-1} a_{i,j}\right) \beta_{j,t} \\
&= \sum_{i=1}^N \left(\bar{\alpha}_{i,t-1} \prod_{s=1}^{t-1} \frac{m_s}{c_s}\right) a_{i,j} \left(\bar{\beta}_{j,t} \frac{1}{c_t} \prod_{s=t+1}^T \frac{m_s}{c_s} \right) \\
&= \sum_{i=1}^N \left(\prod_{s=1}^T \frac{m_s}{c_s}\right) \bar{\alpha}_{i,t-1} a_{i,j} \bar{\beta}_{j,t} \frac{1}{m_t} \\
&= \mathcal{L} \sum_{i=1}^N \bar{\alpha}_{i,t-1} a_{i,j} \bar{\beta}_{j,t} \frac{1}{m_t} \\
\end{align*}
```

Finally, we note that

```math
\frac{\partial \log \mathcal{L}}{\partial \log b_{j,t}} = \frac{\partial \log \mathcal{L}}{\partial b_{j,t}} b_{j,t}
```

To sum up,

```math
\begin{align*}
\frac{\partial \log \mathcal{L}}{\partial \pi_i} &= \frac{b_{i,1}}{m_1} \bar{\beta}_{i,1} \\
\frac{\partial \log \mathcal{L}}{\partial a_{i,j}} &= \sum_{t=1}^{T-1} \bar{\alpha}_{i,t} \frac{b_{j,t+1}}{m_{t+1}} \bar{\beta}_{j,t+1} \\
\frac{\partial \log \mathcal{L}}{\partial \log b_{j,1}} &= \pi_j \frac{b_{j,1}}{m_1} \bar{\beta}_{j,1} = \frac{\bar{\alpha}_{j,1} \bar{\beta}_{j,1}}{c_1} \\
\frac{\partial \log \mathcal{L}}{\partial \log b_{j,t}} &= \sum_{i=1}^N \bar{\alpha}_{i,t-1} a_{i,j} \frac{b_{j,t}}{m_t} \bar{\beta}_{j,t} = \frac{\bar{\alpha}_{j,t} \bar{\beta}_{j,t}}{c_t}
\end{align*}
```