struct HMMBaseImplem <: Implementation end
Base.string(::HMMBaseImplem) = "HMMBase.jl"

function HMMBenchmark.build_model(implem::HMMBaseImplem, instance::Instance, params::Params)
    (; nb_states, obs_dim) = instance
    (; init, trans, means, stds) = params

    a = init
    A = trans
    if obs_dim == 1
        B = [Normal(means[1, i], stds[1, i]) for i in 1:nb_states]
    else
        B = [MvNormal(means[:, i], Diagonal(stds[:, i])) for i in 1:nb_states]
    end

    hmm = HMMBase.HMM(a, A, B)
    return hmm
end

function HMMBenchmark.build_benchmarkables(
    implem::HMMBaseImplem,
    instance::Instance,
    params::Params,
    data::AbstractArray{<:Real,3},
    algos::Vector{String},
)
    (; obs_dim, nb_seqs, bw_iter) = instance
    hmm = build_model(implem, instance, params)

    if obs_dim == 1
        obs_mats = [data[k, :, 1] for k in 1:nb_seqs]
    else
        obs_mats = [data[k, :, :] for k in 1:nb_seqs]
    end
    obs_mat_concat = reduce(vcat, obs_mats)

    benchs = BenchmarkGroup()

    if "forward" in algos
        benchs["forward"] = @benchmarkable begin
            @threads for k in eachindex($obs_mats)
                HMMBase.forward($hmm, $obs_mats[k])
            end
        end evals = 1 samples = 10
    end

    if "viterbi" in algos
        benchs["viterbi"] = @benchmarkable begin
            @threads for k in eachindex($obs_mats)
                HMMBase.viterbi($hmm, $obs_mats[k])
            end
        end evals = 1 samples = 10
    end

    if "forward_backward" in algos
        benchs["forward_backward"] = @benchmarkable begin
            @threads for k in eachindex($obs_mats)
                HMMBase.posteriors($hmm, $obs_mats[k])
            end
        end evals = 1 samples = 10
    end

    if "baum_welch" in algos
        benchs["baum_welch"] = @benchmarkable begin
            HMMBase.fit_mle($hmm, $obs_mat_concat; maxiter=$bw_iter, tol=-Inf)
        end evals = 1 samples = 10
    end

    return benchs
end
