using BenchmarkTools
using Distributions: Normal
using HMMBase: HMMBase
using HiddenMarkovModels

function prepare_models_and_dataset(; N, T)
    p = rand_prob_vec(N)
    A = rand_trans_mat(N)
    dists = [Normal(randn(), 1.0) for i in 1:N]
    hmm = HMM(StandardStateProcess(p, A), StandardObservationProcess(dists))
    hmm_base = HMMBase.HMM(p, A, dists)
    (; state_seq, obs_seq) = rand(hmm, T)
    return (; hmm, hmm_base, obs_seq)
end

SUITE = BenchmarkGroup()

for algo in ["Logdensity", "Viterbi", "Forward-backward", "Baum-Welch"]
    SUITE[algo] = BenchmarkGroup()
    for package in ["HMMs.jl (normal)", "HMMs.jl (log)", "HMMBase.jl"]
        SUITE[algo][package] = BenchmarkGroup()
    end
end

for N in 1:5
    local (; hmm, hmm_base, obs_seq) = prepare_models_and_dataset(; N=N, T=1000)

    ## Logdensity
    SUITE["Logdensity"]["HMMs.jl (normal)"][N] = @benchmarkable logdensityof(
        $hmm, $obs_seq, NormalScale()
    )
    SUITE["Logdensity"]["HMMs.jl (log)"][N] = @benchmarkable logdensityof(
        $hmm, $obs_seq, LogScale()
    )
    SUITE["Logdensity"]["HMMBase.jl"][N] = @benchmarkable HMMBase.forward(
        $hmm_base, $obs_seq
    )

    ## Viterbi
    SUITE["Viterbi"]["HMMs.jl (normal)"][N] = @benchmarkable viterbi(
        $hmm, $obs_seq, NormalScale()
    )
    SUITE["Viterbi"]["HMMs.jl (log)"][N] = @benchmarkable viterbi(
        $hmm, $obs_seq, LogScale()
    )
    SUITE["Viterbi"]["HMMBase.jl"][N] = @benchmarkable HMMBase.viterbi($hmm_base, $obs_seq)

    ## Forward-backward
    SUITE["Baum-Welch"]["HMMs.jl (normal)"][N] = @benchmarkable forward_backward(
        $hmm, $obs_seq, NormalScale()
    )
    SUITE["Baum-Welch"]["HMMs.jl (log)"][N] = @benchmarkable forward_backward(
        $hmm, $obs_seq, LogScale()
    )
    SUITE["Baum-Welch"]["HMMBase.jl"][N] = @benchmarkable HMMBase.posteriors(
        $hmm_base, $obs_seq
    )

    ## Baum-Welch
    SUITE["Baum-Welch"]["HMMs.jl (normal)"][N] = @benchmarkable baum_welch(
        $hmm, $([obs_seq]), NormalScale(); max_iterations=100, rtol=NaN
    )
    SUITE["Baum-Welch"]["HMMs.jl (log)"][N] = @benchmarkable baum_welch(
        $hmm, $([obs_seq]), LogScale(); max_iterations=100, rtol=NaN
    )
    SUITE["Baum-Welch"]["HMMBase.jl"][N] = @benchmarkable HMMBase.fit_mle(
        $hmm_base, $obs_seq; maxiter=100, tol=NaN
    )
end
