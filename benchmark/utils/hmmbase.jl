using BenchmarkTools
using Distributions
using Distributions: PDiagMat
using HMMBase: HMMBase

function rand_params_hmmbase(; N, D)
    p = ones(N) / N
    A = ones(N, N) / N
    μ = randn(N, D)
    σ = 2 * ones(N, D)
    return (; p, A, μ, σ)
end

function rand_model_hmmbase(; N, D)
    (; p, A, μ, σ) = rand_params_hmmbase(; N, D)
    dists = [DiagNormal(μ[n, :], PDiagMat(σ[n, :] .^ 2)) for n in 1:N]
    model = HMMBase.HMM(copy(p), copy(A), dists)
    return model
end

function benchmarkables_hmmbase(; N, D, T, K)
    rand_model_hmmbase(; N, D)
    obs_mat = randn(K * T, D)
    logdensity = @benchmarkable HMMBase.forward(model, $obs_mat) setup = (
        model = rand_model_hmmbase(; N=$N, D=$D)
    )
    viterbi = @benchmarkable HMMBase.viterbi(model, $obs_mat) setup = (
        model = rand_model_hmmbase(; N=$N, D=$D)
    )
    forward_backward = @benchmarkable HMMBase.posteriors(model, $obs_mat) setup = (
        model = rand_model_hmmbase(; N=$N, D=$D)
    )
    baum_welch = @benchmarkable HMMBase.fit_mle(
        model, $obs_mat; maxiter=BAUM_WELCH_ITER, tol=-Inf
    ) setup = (model = rand_model_hmmbase(; N=$N, D=$D))
    return (; logdensity, viterbi, forward_backward, baum_welch)
end
