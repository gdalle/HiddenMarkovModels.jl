using HiddenMarkovModels
using HiddenMarkovModels.Test
using HiddenMarkovModels: PermutedHMM
using Distributions
using Test

## Permuted

perm = [3, 1, 2]
hmm = rand_gaussian_hmm_1d(3)
hmm_perm = PermutedHMM(hmm, perm)

p = initialization(hmm)
A = transition_matrix(hmm)
d = hmm.dists

p_perm = initialization(hmm_perm)
A_perm = transition_matrix(hmm_perm)
d_perm = hmm_perm.dists

@testset "PermutedHMM" begin
    for i in 1:3
        @test p_perm[i] ≈ p[perm[i]]
        @test d_perm[i] ≈ d[perm[i]]
    end
    for i in 1:3, j in 1:3
        @test A_perm[i, j] ≈ A[perm[i], perm[j]]
    end
end
