import numpy as np
import hmmlearn.hmm
import timeit


def create_model(N, D, T, I):
    p = np.random.rand(N)
    p /= p.sum()
    A = np.random.rand(N, N)
    A /= A.sum(1)[:, None]
    mu = np.random.randn(N, D)
    sigma = 2 * np.ones((N, D))

    model = hmmlearn.hmm.GaussianHMM(
        n_components=N,
        covariance_type="diag",
        n_iter=I,
        tol=-np.inf,
        algorithm="viterbi",
        implementation="scaling",
        init_params="",
    )

    model.startprob_ = p
    model.transmat_ = A
    model.means_ = mu
    model.covars_ = sigma**2

    return model


def benchmark(N, D, T, I, number, repeat):
    setup = (
        "model = create_model(N, D, T, I); " + "obs_mat_py = np.random.randn(T, D); "
    )
    logdensity_times = (
        np.array(
            timeit.repeat(
                stmt="model.score(obs_mat_py)",
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
                stmt="model.predict(obs_mat_py)",
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
                stmt="model.predict_proba(obs_mat_py)",
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
                stmt="model.fit(obs_mat_py)",
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
