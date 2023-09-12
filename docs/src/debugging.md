# Debugging

## Numerical overflow

The most frequent error you will encounter is an `OverflowError` during forward-backward, telling you that "some values are infinite / NaN".
This can happen for a variety of reasons, so here are a few leads worth investigating:

* Increase the duration of the sequence / the number of sequences (to get more data)
* Reduce the number of states (to make every one of them useful)
* Add a prior to your transition matrix / observation distributions (to avoid degenerate behavior like zero variance in a Gaussian)
* Pick a better initialization (to start closer to the supposed ground truth)
* Use [LogarithmicNumbers.jl](https://github.com/cjdoris/LogarithmicNumbers.jl) in strategic places (to guarantee numerical stability). Note that these numbers don't play nicely with Distributions.jl, so you may have to roll out your own observation distribution.