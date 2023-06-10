import numpy as np
import hmmlearn.hmm
import timeit


def create_model(N, D, I):
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


def benchmark(N, D, T, K, I, repeat):
    setup = (
        "model = create_model(N, D, I); "
        + "obs_mats_list_py = [np.random.randn(T, D) for k in range(K)]; "
        + "obs_mat_concat_py = np.concatenate(obs_mats_list_py); "
        + "obs_mat_len_py = np.full(K, T); "
    )
    logdensity = timeit.repeat(
        stmt="model.score(obs_mat_concat_py, obs_mat_len_py)",
        setup=setup,
        number=1,
        repeat=repeat,
        globals={**locals(), **globals()},
    )
    viterbi = timeit.repeat(
        stmt="model.predict(obs_mat_concat_py, obs_mat_len_py)",
        setup=setup,
        number=1,
        repeat=repeat,
        globals={**locals(), **globals()},
    )
    forward_backward = timeit.repeat(
        stmt="model.predict_proba(obs_mat_concat_py, obs_mat_len_py)",
        setup=setup,
        number=1,
        repeat=repeat,
        globals={**locals(), **globals()},
    )
    baum_welch = timeit.repeat(
        stmt="model.fit(obs_mat_concat_py, obs_mat_len_py)",
        setup=setup,
        number=1,
        repeat=repeat,
        globals={**locals(), **globals()},
    )
    results = (
        logdensity,
        viterbi,
        forward_backward,
        baum_welch,
    )
    return results
