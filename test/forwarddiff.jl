using FiniteDifferences: FiniteDifferences, central_fdm
using ForwardDiff: ForwardDiff
using HiddenMarkovModels
using HiddenMarkovModels: MyNormal, rand_prob_vec, rand_trans_mat

N = 5

sp = StandardStateProcess(rand_prob_vec(N), rand_trans_mat(N))
μ = randn(N)
op = StandardObservationProcess([MyNormal(μ[i], 1.0) for i in 1:N])
hmm = HMM(sp, op)

(; state_seq, obs_seq) = rand(hmm, 100);

function logdensity_μ(μ)
    op = StandardObservationProcess([MyNormal(μ[i], 1.0) for i in 1:N])
    hmm = HMM(sp, op)
    return logdensityof(hmm, obs_seq)
end

g1 = ForwardDiff.gradient(logdensity_μ, μ)
g2 = FiniteDifferences.grad(central_fdm(5, 1), logdensity_μ, μ)[1]
@test isapprox(g1, g2)
