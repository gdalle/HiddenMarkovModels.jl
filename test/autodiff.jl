using Distributions
using FiniteDifferences: FiniteDifferences, central_fdm
using ForwardDiff: ForwardDiff
using HiddenMarkovModels
using SimpleUnPack
using Test
using Zygote: Zygote

N = 5
T = 100

p = rand_prob_vec(N)
A = rand_trans_mat(N)
μ = rand(N)
d = [Normal(μ[i], 1.0) for i in 1:N]
hmm = HMM(p, A, d)

obs_seq = rand(hmm, T).obs_seq;

function f_init(_p)
    hmm = HMM(_p, A, d)
    return logdensityof(hmm, obs_seq)
end

g1 = ForwardDiff.gradient(f_init, p)
g2 = Zygote.gradient(f_init, p)[1]
@test isapprox(g1, g2)

function f_trans(_A)
    hmm = HMM(p, _A, d)
    return logdensityof(hmm, obs_seq)
end

g1 = ForwardDiff.gradient(f_trans, A)
g2 = Zygote.gradient(f_trans, A)[1]
@test isapprox(g1, g2)

function f_d(_μ)
    hmm = HMM(p, A, [Normal(_μ[i], 1.0) for i in 1:N])
    return logdensityof(hmm, obs_seq)
end

g0 = FiniteDifferences.grad(central_fdm(5, 1), f_d, μ)[1]
g1 = ForwardDiff.gradient(f_d, μ)
g2 = Zygote.gradient(f_d, μ)[1]
@test isapprox(g0, g1)
@test isapprox(g0, g2)
