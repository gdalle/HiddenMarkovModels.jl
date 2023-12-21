module HMMBenchmarkHMMBaseExt

using BenchmarkTools
using Distributions
using HMMBase
using SparseArrays

function benchmarkables_hmmbase(; configuration, algos)
    @unpack sparse, nb_states, obs_dim, seq_length, nb_seqs, bw_iter = configuration

    # Model
    a = ones(nb_states) / nb_states
    A = ones(nb_states, nb_states) / nb_states
    if obs_dim == 1
        B = [Normal(i, 1.0) for _ in 1:nb_states]
    else
        B = [MvNormal(i .* ones(obs_dim), Diagonal(ones(obs_dim))) for _ in 1:nb_states]
    end
    hmm = HMMBase.HMM(a, A, B)

    obs_mat = rand(hmm, seq_length * nb_seqs)

    # Benchmarks
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
            HMMBase.fit_mle(model, $obs_mat; maxiter=$I, tol=-Inf)
        end setup = (model = rand_model_hmmbase(; N=$N, D=$D))
    end

    return benchs
end

end
