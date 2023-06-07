using Distributions: Normal
using HiddenMarkovModels
using HiddenMarkovModels: MyNormal, sum_to_one!
using JET
using SparseArrays: sprand, nnz
using Test

function test_type_stability(hmm; T)
    (; state_seq, obs_seq) = rand(hmm, T)

    @testset "Logdensity" begin
        for scale in (NormalScale(), LogScale())
            @inferred logdensityof(hmm, obs_seq, scale)
            @test_opt target_modules = (HMMs,) logdensityof(hmm, obs_seq, scale)
            @test_call logdensityof(hmm, obs_seq, scale)
        end
    end

    @testset "Viterbi" begin
        for scale in (NormalScale(), LogScale())
            @inferred viterbi(hmm, obs_seq, scale)
            @test_opt target_modules = (HMMs,) viterbi(hmm, obs_seq, scale)
            @test_call viterbi(hmm, obs_seq, scale)
        end
    end

    @testset "Forward-backward" begin
        for scale in (NormalScale(), LogScale())
            @inferred forward_backward(hmm, obs_seq, scale)
            @test_opt target_modules = (HMMs,) forward_backward(hmm, obs_seq, scale)
            @test_call forward_backward(hmm, obs_seq, scale)
        end
    end

    @testset "Baum-Welch" begin
        for scale in (NormalScale(), LogScale())
            @inferred baum_welch(hmm, [obs_seq], scale)
            @test_opt target_modules = (HMMs,) baum_welch(hmm, [obs_seq], scale)
            @test_call baum_welch(hmm, [obs_seq], scale)
        end
    end
end

function test_sparsity(hmm; T)
    (; state_seq, obs_seq) = rand(hmm, T)
    hmm_est, logL_evolution = baum_welch(hmm, [obs_seq])
    @test nnz(initial_distribution(hmm_est.state_process)) <=
        nnz(initial_distribution(hmm.state_process))
    @test nnz(transition_matrix(hmm_est.state_process)) <=
        nnz(transition_matrix(hmm.state_process))
end

N = 10

sp = StandardStateProcess(rand_prob_vec(N), rand_trans_mat(N))
op = StandardObservationProcess([Normal(randn(), 1.0) for i in 1:N])
hmm = HMM(sp, op)

p_sparse = sprand(N, 0.5);
sum_to_one!(p_sparse);
A_sparse = sprand(N, N, 0.5);
A_sparse[:, end] .+= 1;
foreach(sum_to_one!, eachrow(A_sparse))
sp_sparse = StandardStateProcess(p_sparse, A_sparse)
op_sparse = StandardObservationProcess([MyNormal(randn(), 1.0) for i in 1:N])
hmm_sparse = HMM(sp_sparse, op_sparse)

@testset verbose = true "Dense" begin
    test_type_stability(hmm; T=100)
end

@testset verbose = true "Sparse" begin
    test_type_stability(hmm_sparse; T=100)
    test_sparsity(hmm_sparse; T=100)
end
