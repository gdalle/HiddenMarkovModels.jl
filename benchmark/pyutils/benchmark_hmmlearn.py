import numpy as np
import hmmlearn.hmm
import timeit


def create_model(N, D, max_iterations):
    p = np.random.rand(N)
    p /= p.sum()
    A = np.random.rand(N, N)
    A /= A.sum(1)[:, None]
    mu = np.random.randn(N, D)
    sigma = np.ones((N, D))

    model = hmmlearn.hmm.GaussianHMM(
        n_components=N,
        covariance_type="diag",
        n_iter=max_iterations,
        tol=-np.inf,
        algorithm="viterbi",
        implementation="scaling",
        init_params="",
    )

    model.startprob_ = p
    model.transmat_ = A
    model.means_ = mu
    model.covars_ = sigma

    return model


def benchmark(N, D, T, max_iterations, number=5, repeat=5):
    setup_inference = "model = create_model(N, D, max_iterations); obs_mat, _ = model.sample(T)"
    setup_learning = setup_inference + "; model_init = create_model(N, D, max_iterations)"
    logdensity_times = np.array(timeit.repeat(
        "model.score(obs_mat)",
        setup=setup_inference,
        number=number,
        repeat=repeat,
        globals={**locals(), **globals()}
    )) / number
    viterbi_times = np.array(timeit.repeat(
        "model.predict(obs_mat)",
        setup=setup_inference,
        number=number,
        repeat=repeat,
        globals={**locals(), **globals()}
    )) / number
    forward_backward_times = np.array(timeit.repeat(
        "model.predict_proba(obs_mat)",
        setup=setup_inference,
        number=number,
        repeat=repeat,
        globals={**locals(), **globals()}
    )) / number
    baum_welch_times = np.array(timeit.repeat(
        "model_init.fit(obs_mat)",
        setup=setup_learning,
        number=number,
        repeat=repeat,
        globals={**locals(), **globals()}
    )) / number
    return logdensity_times, viterbi_times, forward_backward_times, baum_welch_times