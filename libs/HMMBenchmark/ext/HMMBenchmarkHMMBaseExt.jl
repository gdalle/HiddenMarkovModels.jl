module HMMBenchmarkHMMBaseExt

using BenchmarkTools
using Distributions
using HMMBase
using HMMBenchmark
using Random: AbstractRNG
using SparseArrays

function HMMBenchmark.benchmarkables_hmmbase(rng::AbstractRNG; configuration, algos)
    (; sparse, nb_states, obs_dim, seq_length, nb_seqs, bw_iter) = configuration

    # Model
    a = ones(nb_states) / nb_states
    A = ones(nb_states, nb_states) / nb_states
    if obs_dim == 1
        B = [Normal(i, 1.0) for i in 1:nb_states]
    else
        B = [MvNormal(i .* ones(obs_dim), Diagonal(ones(obs_dim))) for i in 1:nb_states]
    end
    hmm = HMMBase.HMM(a, A, B)

    # Data
    obs_mat = rand(rng, hmm, seq_length * nb_seqs)  # concat insted of multiple sequences

    # Benchmarks
    benchs = Dict()

    if "logdensity" in algos
        benchs["logdensity"] = @benchmarkable begin
            HMMBase.forward($hmm, $obs_mat)
        end
    end

    if "forward" in algos
        benchs["forward"] = @benchmarkable begin
            HMMBase.forward($hmm, $obs_mat)
        end
    end

    if "viterbi" in algos
        benchs["viterbi"] = @benchmarkable begin
            HMMBase.viterbi($hmm, $obs_mat)
        end
    end

    if "forward_backward" in algos
        benchs["forward_backward"] = @benchmarkable begin
            HMMBase.posteriors($hmm, $obs_mat)
        end
    end

    if "baum_welch" in algos
        benchs["baum_welch"] = @benchmarkable begin
            HMMBase.fit_mle($hmm, $obs_mat; maxiter=$bw_iter, tol=-Inf)
        end
    end

    return benchs
end

end
