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

function rand_model_pomegranate(; N, D, I)
    (; p, A, μ, σ) = rand_params_pomegranate(; N, D)
    distributions = pylist([
        pomegranate.distributions.Normal(;
            means=μ[n], covs=torch.square(σ[n]), covariance_type="diag"
        ) for n in 0:(N - 1)
    ])
    model = pomegranate.hmm.DenseHMM(;
        distributions=distributions, edges=A, starts=p, max_iter=I, tol=1e-10, verbose=false
    )
    return model
end

function benchmarkables_pomegranate(; N, D, T, K, I)
    rand_model_pomegranate(; N, D, I)
    obs_tens_py = torch.randn(K, T, D)
    logdensity = @benchmarkable pycall(model_forward, $obs_tens_py) setup = (
        model_forward = rand_model_pomegranate(; N=$N, D=$D, I=$I).forward
    )
    viterbi = @benchmarkable pycall(model_predict, $obs_tens_py) setup = (
        model_predict = rand_model_pomegranate(; N=$N, D=$D, I=$I).predict
    )
    forward_backward = @benchmarkable pycall(model_forward_backward, $obs_tens_py) setup = (
        model_forward_backward = rand_model_pomegranate(; N=$N, D=$D, I=$I).forward_backward
    )
    baum_welch = @benchmarkable pycall(model_fit, $obs_tens_py) setup = (
        model_fit = rand_model_pomegranate(; N=$N, D=$D, I=$I).fit
    )
    return (; logdensity, viterbi, forward_backward, baum_welch)
end
