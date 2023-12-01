using Distributions
using FiniteDifferences: FiniteDifferences, central_fdm
using ForwardDiff: ForwardDiff
using HiddenMarkovModels
using SimpleUnPack
using Test
using Zygote: Zygote

N = 5
T = 10

init = rand_prob_vec(N)
trans = rand_trans_mat(N)
means = rand(N)
dists = [Normal(means[i], 1.0) for i in 1:N]
hmm = HMM(init, trans, dists)

obs_seq = rand(hmm, T).obs_seq;

function f_init(_init)
    new_hmm = HMM(_init, trans, dists)
    return logdensityof(new_hmm, obs_seq)
end

g1 = ForwardDiff.gradient(f_init, init)
g2 = Zygote.gradient(f_init, init)[1]
@test isapprox(g1, g2)

function f_trans(_trans)
    new_hmm = HMM(init, _trans, dists)
    return logdensityof(new_hmm, obs_seq)
end

g1 = ForwardDiff.gradient(f_trans, trans)
g2 = Zygote.gradient(f_trans, trans)[1]
@test isapprox(g1, g2)

function f_means(_means)
    new_hmm = HMM(init, trans, [Normal(_means[i], 1.0) for i in 1:N])
    return logdensityof(new_hmm, obs_seq)
end

g0 = FiniteDifferences.grad(central_fdm(5, 1), f_means, means)[1]
g1 = ForwardDiff.gradient(f_means, means)
g2 = Zygote.gradient(f_means, means)[1]
@test isapprox(g0, g1)
@test isapprox(g0, g2)
