
function rand_params_hmmbase(; N, D)
    p = ones(N) / N
    A = ones(N, N) / N
    μ = randn(N, D)
    σ = 2 * ones(N, D)
    return (; p, A, μ, σ)
end

function rand_model_hmmbase(; N, D)
    @unpack p, A, μ, σ = rand_params_hmmbase(; N, D)
    if D == 1
        dists = [Normal(μ[n, 1], σ[n, 1]) for n in 1:N]
    else
        dists = [DiagNormal(μ[n, :], PDiagMat(σ[n, :] .^ 2)) for n in 1:N]
    end
    model = HMMBase.HMM(p, A, dists)
    return model
end

function benchmarkables_hmmbase(; algos, N, D, T, K, I)
    rand_model_hmmbase(; N, D)
    if D == 1
        obs_mat = randn(K * T)
    else
        obs_mat = randn(K * T, D)
    end
    benchs = Dict()
    if "logdensity" in algos
        benchs["logdensity"] = @benchmarkable HMMBase.forward(model, $obs_mat) setup = (
            model = rand_model_hmmbase(; N=$N, D=$D)
        )
    end
    if "viterbi" in algos
        benchs["viterbi"] = @benchmarkable HMMBase.viterbi(model, $obs_mat) setup = (
            model = rand_model_hmmbase(; N=$N, D=$D)
        )
    end
    if "forward_backward" in algos
        benchs["forward_backward"] = @benchmarkable HMMBase.posteriors(model, $obs_mat) setup = (
            model = rand_model_hmmbase(; N=$N, D=$D)
        )
    end
    if "baum_welch" in algos
        benchs["baum_welch"] = @benchmarkable HMMBase.fit_mle(
            model, $obs_mat; maxiter=$I, tol=-Inf
        ) setup = (model = rand_model_hmmbase(; N=$N, D=$D))
    end
    return benchs
end
