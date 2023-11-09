using Distributions
using HiddenMarkovModels
using HiddenMarkovModels: sum_to_one!
using LinearAlgebra
using SparseArrays
using StaticArrays
using SimpleUnPack
using Test

N, T = 3, 1000

## Sparse

p = ones(N) / N;
A = SparseMatrixCSC(SymTridiagonal(ones(N), ones(N - 1)));
foreach(sum_to_one!, eachrow(A));
d = [Normal(i + randn(), 1.0) for i in 1:N];
d_init = [Normal(i + randn(), 1.0) for i in 1:N];

hmm = HMM(p, A, d);
hmm_init = HMM(p, A, d_init);

obs_seq = rand(hmm, T).obs_seq;

γ, ξ, logL = forward_backward(hmm, obs_seq);
hmm_est, logL_evolution = @inferred baum_welch(hmm_init, obs_seq);

@testset "Sparse" begin
    @test eltype(ξ) <: AbstractSparseArray
    @test typeof(hmm_est) == typeof(hmm_init)
    @test nnz(transition_matrix(hmm_est)) <= nnz(transition_matrix(hmm))
end

## Static

p = MVector{N}(ones(N) / N);
A = MMatrix{N,N}(rand_trans_mat(N));
d = MVector{N}([Normal(randn(), 1.0) for i in 1:N]);
d_init = MVector{N}([Normal(randn(), 1.0) for i in 1:N]);

hmm = HMM(p, A, d);
hmm_init = HMM(p, A, d_init);
obs_seq = rand(hmm, T).obs_seq;

γ, ξ, logL = forward_backward(hmm, obs_seq);
hmm_est, logL_evolution = @inferred baum_welch(hmm_init, obs_seq);

@testset "Static" begin
    @test eltype(ξ) <: StaticArray
    @test typeof(hmm_est) == typeof(hmm_init)
end
