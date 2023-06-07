using BenchmarkTools
using Distributions: Normal
using HMMBase: HMMBase
using HiddenMarkovModels
using HiddenMarkovModels: MyNormal, rand_prob_vec, rand_trans_mat
using LogarithmicNumbers: ULogarithmic

function prepare_models_and_dataset(; N, T)
    p = rand_prob_vec(N)
    A = rand_trans_mat(N)
    dists = [Normal(randn(), 1.0) for i in 1:N]
    dists_custom = [MyNormal(randn(), 1.0) for i in 1:N]
    dists_custom_log = [MyNormal(randn(), ULogarithmic(1.0)) for i in 1:N]
    hmm = HMM(StandardStateProcess(p, A), StandardObservationProcess(dists_custom))
    hmm_log = HMM(StandardStateProcess(p, A), StandardObservationProcess(dists_custom_log))
    hmm_base = HMMBase.HMM(p, A, dists)
    (; state_seq, obs_seq) = rand(hmm, T)
    return (; hmm, hmm_log, hmm_base, obs_seq)
end

SUITE = BenchmarkGroup()

for algo in ["Forward", "Viterbi", "Baum-Welch"]
    SUITE[algo] = BenchmarkGroup()
    for package in
        ["HiddenMarkovModels.jl (plain)", "HiddenMarkovModels.jl (log)", "HMMBase.jl"]
        SUITE[algo][package] = BenchmarkGroup()
    end
end

for N in 1:5
    local (; hmm, hmm_log, hmm_base, obs_seq) = prepare_models_and_dataset(; N=N, T=1000)
    ## Baum-Welch
    SUITE["Baum-Welch"]["HiddenMarkovModels.jl (plain)"][N] = @benchmarkable baum_welch(
        $hmm, $([obs_seq]); max_iterations=100, rtol=NaN
    )
    SUITE["Baum-Welch"]["HiddenMarkovModels.jl (log)"][N] = @benchmarkable baum_welch(
        $hmm_log, $([obs_seq]); max_iterations=100, rtol=NaN
    )
    SUITE["Baum-Welch"]["HMMBase.jl"][N] = @benchmarkable HMMBase.fit_mle(
        $hmm_base, $obs_seq; maxiter=100, tol=NaN
    )
end
