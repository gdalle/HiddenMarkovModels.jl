import numpy as np
import torch
import pomegranate.hmm
import pomegranate.distributions
import timeit


def create_model(N, D, T, I):
    p = torch.rand(N)
    p /= p.sum()
    A = torch.rand(N, N)
    A /= A.sum(1)[:, None]
    mu = torch.randn(N, D)
    sigma = 2 * torch.ones((N, D))

    distributions = [
        pomegranate.distributions.Normal(
            means=mu[n, :],
            covs=sigma[n, :] ** 2,
            covariance_type="diag",
        )
        for n in range(N)
    ]

    model = pomegranate.hmm.DenseHMM(
        distributions=distributions,
        edges=A,
        starts=p,
        sample_length=T,
        max_iter=I,
        tol=1e-10,
        verbose=False,
    )
    return model


def benchmark(N, D, T, I, number, repeat):
    setup = (
        "model = create_model(N, D, T, I); " + "obs_tens_py = torch.randn(1, T, D); "
    )
    logdensity_times = (
        np.array(
            timeit.repeat(
                stmt="model.forward(obs_tens_py)",
                setup=setup,
                number=number,
                repeat=repeat,
                globals={**locals(), **globals()},
            )
        )
        / number
    )
    viterbi_times = (
        np.array(
            timeit.repeat(
                stmt="model.predict(obs_tens_py)",
                setup=setup,
                number=number,
                repeat=repeat,
                globals={**locals(), **globals()},
            )
        )
        / number
    )
    forward_backward_times = (
        np.array(
            timeit.repeat(
                stmt="model.forward_backward(obs_tens_py)",
                setup=setup,
                number=number,
                repeat=repeat,
                globals={**locals(), **globals()},
            )
        )
        / number
    )
    baum_welch_times = (
        np.array(
            timeit.repeat(
                stmt="model.fit(obs_tens_py)",
                setup=setup,
                number=number,
                repeat=repeat,
                globals={**locals(), **globals()},
            )
        )
        / number
    )
    results = (
        logdensity_times,
        viterbi_times,
        forward_backward_times,
        baum_welch_times,
    )
    return results
