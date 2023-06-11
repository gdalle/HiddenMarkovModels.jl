using BenchmarkTools
using HiddenMarkovModels: HMMs

function rand_params_hmms(; N, D)
    p = ones(N) / N
    A = ones(N, N) / N
    μ = randn(N, D)
    σ = 2 * ones(N, D)
    return (; p, A, μ, σ)
end

function rand_model_hmms(; N, D)
    (; p, A, μ, σ) = rand_params_hmms(; N, D)
    if D == 1
        dists = [Normal(μ[n, 1], σ[n, 1]) for n in 1:N]
    else
        dists = [HMMs.LightDiagNormal(μ[n, :], σ[n, :]) for n in 1:N]
    end
    model = HMMs.HMM(p, A, dists)
    return model
end

function benchmarkables_hmms(; N, D, T, K)
    rand_model_hmms(; N, D)
    if D == 1
        obs_seqs = [[randn() for t in 1:T] for k in 1:K]
    else
        obs_seqs = [[randn(D) for t in 1:T] for k in 1:K]
    end
    logdensity = @benchmarkable HMMs.logdensityof(model, $obs_seqs, $K) setup = (
        model = rand_model_hmms(; N=$N, D=$D)
    )
    viterbi = @benchmarkable HMMs.viterbi(model, $obs_seqs, $K) setup = (
        model = rand_model_hmms(; N=$N, D=$D)
    )
    forward_backward = @benchmarkable HMMs.forward_backward(model, $obs_seqs, $K) setup = (
        model = rand_model_hmms(; N=$N, D=$D)
    )
    baum_welch = @benchmarkable HMMs.baum_welch(
        model, $obs_seqs, $K; max_iterations=BAUM_WELCH_ITER, rtol=-Inf
    ) setup = (model = rand_model_hmms(; N=$N, D=$D))
    return (; logdensity, viterbi, forward_backward, baum_welch)
end
