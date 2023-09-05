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
dists = [Normal(μ[i], 1.0) for i in 1:N]
hmm = HMM(p, A, dists)

@unpack state_seq, obs_seq = rand(hmm, T);

function f_init(_p)
    hmm = HMM(_p, A, dists)
    return logdensityof(hmm, obs_seq)
end

g1 = ForwardDiff.gradient(f_init, p)
@test_broken Zygote.gradient(f_init, p)[1]

function f_trans(_A)
    hmm = HMM(p, _A, dists)
    return logdensityof(hmm, obs_seq)
end

g1 = ForwardDiff.gradient(f_trans, A)
@test_broken Zygote.gradient(f_trans, A)[1]

function f_dists(_μ)
    hmm = HMM(p, A, [Normal(_μ[i], 1.0) for i in 1:N])
    return logdensityof(hmm, obs_seq)
end

g0 = FiniteDifferences.grad(central_fdm(5, 1), f_dists, μ)[1]
g1 = ForwardDiff.gradient(f_dists, μ)
@test_broken Zygote.gradient(f_dists, μ)[1]
@test isapprox(g0, g1)
