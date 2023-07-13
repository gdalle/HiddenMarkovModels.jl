using Distributions
using FiniteDifferences: FiniteDifferences, central_fdm
using ForwardDiff: ForwardDiff
using HiddenMarkovModels
using SimpleUnPack
using Test

N = 5
T = 100

p = rand_prob_vec(N)
A = rand_trans_mat(N)
μ = rand(N)
dists = [Normal(μ[i], 1.0) for i in 1:N]
hmm = HMM(p, A, dists)

@unpack state_seq, obs_seq = rand(hmm, T);

function f(μ)
    new_dists = [Normal(μ[i], 1.0) for i in 1:N]
    hmm = HMM(p, A, new_dists)
    return logdensityof(hmm, obs_seq)
end

g1 = ForwardDiff.gradient(f, μ)
g2 = FiniteDifferences.grad(central_fdm(5, 1), f, μ)[1]
@test isapprox(g1, g2)
