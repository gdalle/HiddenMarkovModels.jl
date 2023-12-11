module HMMBenchmarkHMMBaseExt

using BenchmarkTools
using Distributions
using HMMBase
using SparseArrays

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
        obs_mats = [randn(T) for k in 1:K]
    else
        obs_mats = [randn(T, D) for k in 1:K]
    end
    obs_mats_concat = randn(K * T, D)
    benchs = Dict()
    if "logdensity" in algos
        benchs["logdensity"] = @benchmarkable begin
            for k in 1:($K)
                HMMBase.forward(model, $obs_mats[k])
            end
        end setup = (model = rand_model_hmmbase(; N=$N, D=$D))
    end
    if "viterbi" in algos
        benchs["viterbi"] = @benchmarkable begin
            for k in 1:($K)
                HMMBase.viterbi(model, $obs_mats[k])
            end
        end setup = (model = rand_model_hmmbase(; N=$N, D=$D))
    end
    if "forward_backward" in algos
        benchs["forward_backward"] = @benchmarkable begin
            for k in 1:($K)
                HMMBase.posteriors(model, $obs_mats[k])
            end
        end setup = (model = rand_model_hmmbase(; N=$N, D=$D))
    end
    if "baum_welch" in algos
        benchs["baum_welch"] = @benchmarkable begin
            HMMBase.fit_mle(model, $obs_mats_concat; maxiter=$I, tol=-Inf)
        end setup = (model = rand_model_hmmbase(; N=$N, D=$D))
    end
    return benchs
end

end
