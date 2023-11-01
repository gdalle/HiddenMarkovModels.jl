using HiddenMarkovModels
using Distributions
using Test

p = rand_prob_vec(3)
A = rand_trans_mat(3)
dists = [Normal(i) for i in 1:3]

hmm = HMM(p, A, dists)
perm = [3, 1, 2]

hmm_perm = PermutedHMM(hmm, perm)
p_perm = initialization(hmm_perm)
A_perm = transition_matrix(hmm_perm)
dists_perm = [obs_distribution(hmm_perm, i) for i in 1:3]

for i in 1:3
    @test p_perm[i] ≈ p[perm[i]]
    @test dists_perm[i] ≈ dists[perm[i]]
end

for i in 1:3, j in 1:3
    @test A_perm[i, j] ≈ A[perm[i], perm[j]]
end
