using Distributions: Normal
using FiniteDifferences: FiniteDifferences, central_fdm
using ForwardDiff: ForwardDiff
using HiddenMarkovModels
using Test

N = 5

sp = StandardStateProcess(rand_prob_vec(N), rand_trans_mat(N))
μ = randn(N)
op = StandardObservationProcess([Normal(μ[i], 1.0) for i in 1:N])
hmm = HMM(sp, op)

(; state_seq, obs_seq) = rand(hmm, 100);

function f(μ, scale)
    op = StandardObservationProcess([Normal(μ[i], 1.0) for i in 1:N])
    hmm = HMM(sp, op)
    return logdensityof(hmm, obs_seq, scale)
end

g1 = ForwardDiff.gradient(_μ -> f(_μ, NormalScale()), μ)
g1_log = ForwardDiff.gradient(_μ -> f(_μ, LogScale()), μ)
g2 = FiniteDifferences.grad(central_fdm(5, 1), _μ -> f(_μ, LogScale()), μ)[1]
@test isapprox(g1, g2)
@test isapprox(g1_log, g2)
