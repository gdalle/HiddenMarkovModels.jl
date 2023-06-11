using BenchmarkTools
using PythonCall

torch = pyimport("torch")
pomegranate = pyimport("pomegranate")
pyimport("pomegranate.hmm")
pyimport("pomegranate.distributions")

function rand_params_pomegranate(; N, D)
    p = torch.ones(N) / N
    A = torch.ones(N, N) / N
    μ = torch.randn(N, D)
    σ = 2 * torch.ones(N, D)
    return (; p, A, μ, σ)
end

function rand_model_pomegranate(; N, D)
    (; p, A, μ, σ) = rand_params_pomegranate(; N, D)
    distributions = pylist([
        pomegranate.distributions.Normal(;
            means=μ[n], covs=torch.square(σ[n]), covariance_type="diag"
        ) for n in 0:(N - 1)
    ])
    model = pomegranate.hmm.DenseHMM(;
        distributions=distributions,
        edges=A,
        starts=p,
        max_iter=BAUM_WELCH_ITER,
        tol=1e-10,
        verbose=false,
    )
    return model
end

function benchmarkables_pomegranate(; algos, N, D, T, K)
    rand_model_pomegranate(; N, D)
    obs_tens_py = torch.randn(K, T, D)
    bench = Dict()
    if "logdensity" in algos
        bench["logdensity"] = @benchmarkable pycall(model_forward, $obs_tens_py) setup = (
            model_forward = rand_model_pomegranate(; N=$N, D=$D).forward
        )
    end
    if "forward_backward" in algos
        bench["forward_backward"] = @benchmarkable pycall(
            model_forward_backward, $obs_tens_py
        ) setup = (
            model_forward_backward = rand_model_pomegranate(; N=$N, D=$D).forward_backward
        )
    end
    if "baum_welch" in algos
        bench["baum_welch"] = @benchmarkable pycall(model_fit, $obs_tens_py) setup = (
            model_fit = rand_model_pomegranate(; N=$N, D=$D).fit
        )
    end
    return bench
end
