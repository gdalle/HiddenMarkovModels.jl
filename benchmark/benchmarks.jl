using BenchmarkTools
using Distributions: Normal
using HMMBase: HMMBase
using HiddenMarkovModels

function prepare_models_and_dataset(; N, T)
    p = rand_prob_vec(N)
    A = rand_trans_mat(N)
    dists = [Normal(randn(), 1.0) for i in 1:N]
    hmm = HMM(StandardStateProcess(p, A), StandardObservationProcess(dists))
    hmm_base = HMMBase.HMM(copy(p), copy(A), copy(dists))
    (; state_seq, obs_seq) = rand(hmm, T)
    return (; hmm, hmm_base, obs_seq)
end

function define_suite(; N_values, T, baum_welch_iterations)
    SUITE = BenchmarkGroup()

    for algo in ["Logdensity", "Viterbi", "Forward-backward", "Baum-Welch"]
        SUITE[algo] = BenchmarkGroup()
        SUITE[algo]["HMMBase.jl"] = BenchmarkGroup()
        SUITE[algo]["HiddenMarkovModels.jl"] = BenchmarkGroup()
    end

    for N in N_values
        local (; hmm, hmm_base, obs_seq) = prepare_models_and_dataset(; N=N, T=T)

        implem = "HMMBase.jl"
        SUITE["Logdensity"][implem][N] = @benchmarkable HMMBase.forward($hmm_base, $obs_seq)
        SUITE["Viterbi"][implem][N] = @benchmarkable HMMBase.viterbi($hmm_base, $obs_seq)
        SUITE["Forward-backward"][implem][N] = @benchmarkable HMMBase.posteriors(
            $hmm_base, $obs_seq
        )
        SUITE["Baum-Welch"][implem][N] = @benchmarkable HMMBase.fit_mle(
            $hmm_base, $obs_seq; maxiter=baum_welch_iterations, tol=NaN
        )

        implem = "HiddenMarkovModels.jl"
        SUITE["Logdensity"][implem][N] = @benchmarkable logdensityof($hmm, $obs_seq)
        SUITE["Viterbi"][implem][N] = @benchmarkable viterbi($hmm, $obs_seq)
        SUITE["Forward-backward"][implem][N] = @benchmarkable forward_backward(
            $hmm, $obs_seq
        )
        SUITE["Baum-Welch"][implem][N] = @benchmarkable baum_welch(
            $hmm, $([obs_seq]); max_iterations=$baum_welch_iterations, rtol=NaN
        )
    end

    return SUITE
end

SUITE = define_suite(; N_values=2:2:10, T=100, baum_welch_iterations=10)
