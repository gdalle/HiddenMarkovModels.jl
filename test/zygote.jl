using FiniteDifferences: FiniteDifferences, central_fdm
using Distributions: Normal
using HiddenMarkovModels
using HiddenMarkovModels: rand_prob_vec, rand_trans_mat
using Zygote: Zygote

N = 10

p = rand_prob_vec(N);
A = rand_trans_mat(N);
sp = StandardStateProcess(p, A)
μ = float.(1:N)
op = StandardObservationProcess([Normal(μ[i], 1.0) for i in 1:N])
hmm = HMM(sp, op)

(; state_seq, obs_seq) = rand(hmm, 100);

function logdensity_μ(μ)
    op = StandardObservationProcess([Normal(μ[i], 1.0) for i in 1:N])
    hmm = HMM(sp, op)
    return logdensityof(hmm, obs_seq)
end

# g1 = Zygote.gradient(logdensity_μ, μ)[1]  # mutation needs custom chain rule
# g2 = FiniteDifferences.grad(central_fdm(5, 1), logdensity_μ, μ)[1]
# @test isapprox(g1, g2)
