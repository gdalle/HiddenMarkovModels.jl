using HiddenMarkovModels
using HiddenMarkovModels: rand_prob_vec, rand_trans_mat
using Distributions
using Test

@testset "Allow NaN density" begin
    init = rand_prob_vec(2)
    trans = rand_trans_mat(2)
    dists = [Normal(i, Inf) for i in 1:2]
    hmm = HMM(init, trans, dists)
    obs_seq = rand(5)
    @test isnan(logdensityof(hmm, obs_seq))
end
