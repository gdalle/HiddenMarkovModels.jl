function rand_params_hmms(; N, D)
    p = ones(N) / N
    A = ones(N, N) / N
    μ = randn(N, D)
    σ = 2 * ones(N, D)
    return (; p, A, μ, σ)
end

function rand_model_hmms(; N, D)
    @unpack p, A, μ, σ = rand_params_hmms(; N, D)
    if D == 1
        dists = [Normal(μ[n, 1], σ[n, 1]) for n in 1:N]
    else
        dists = [HiddenMarkovModels.LightDiagNormal(μ[n, :], σ[n, :]) for n in 1:N]
    end
    model = HiddenMarkovModels.HMM(p, A, dists)
    return model
end

function benchmarkables_hmms(; algos, N, D, T, K, I)
    rand_model_hmms(; N, D)
    if D == 1
        obs_seqs = [[randn() for t in 1:T] for k in 1:K]
    else
        obs_seqs = [[randn(D) for t in 1:T] for k in 1:K]
    end
    benchs = Dict()
    if "logdensity" in algos
        benchs["logdensity"] = @benchmarkable begin
            for k in 1:($K)
                HiddenMarkovModels.logdensityof(model, $obs_seqs[k])
            end
        end setup = (model = rand_model_hmms(; N=$N, D=$D))
    end
    if "viterbi" in algos
        benchs["viterbi"] = @benchmarkable begin
            for k in 1:($K)
                HiddenMarkovModels.viterbi(model, $obs_seqs[k])
            end
        end setup = (model = rand_model_hmms(; N=$N, D=$D))
    end
    if "forward_backward" in algos
        benchs["forward_backward"] = @benchmarkable begin
            for k in 1:($K)
                HiddenMarkovModels.forward_backward(model, $obs_seqs[k])
            end
        end setup = (model = rand_model_hmms(; N=$N, D=$D))
    end
    if "baum_welch" in algos
        benchs["baum_welch"] = @benchmarkable begin
            HiddenMarkovModels.baum_welch(
                model, $obs_seqs, $K; max_iterations=$I, atol=-Inf
            )
        end setup = (model = rand_model_hmms(; N=$N, D=$D))
    end
    return benchs
end
