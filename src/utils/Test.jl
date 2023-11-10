module Test

using Distributions
using Distributions: PDiagMat
using HiddenMarkovModels
using HiddenMarkovModels: LightDiagNormal, sum_to_one!
using LinearAlgebra
using SparseArrays

export rand_categorical_hmm
export rand_gaussian_hmm_1d
export rand_gaussian_hmm_2d
export rand_gaussian_hmm_2d_light

function sparse_trans_mat(N)
    A = sparse(SymTridiagonal(ones(N), ones(N - 1)))
    foreach(sum_to_one!, eachrow(A))
    return A
end

function rand_categorical_hmm(N, D; sparse_trans=false)
    p = ones(N) / N
    A = sparse_trans ? sparse_trans_mat(N) : rand_trans_mat(N)
    d = [Categorical(rand_prob_vec(D)) for i in 1:N]
    return HMM(p, A, d)
end

function rand_gaussian_hmm_1d(N; sparse_trans=false)
    p = ones(N) / N
    A = sparse_trans ? sparse_trans_mat(N) : rand_trans_mat(N)
    d = [Normal(randn(), 1) for i in 1:N]
    return HMM(p, A, d)
end

function rand_gaussian_hmm_2d(N, D; sparse_trans=false)
    p = ones(N) / N
    A = sparse_trans ? sparse_trans_mat(N) : rand_trans_mat(N)
    d = [DiagNormal(randn(D), PDiagMat(ones(D) .^ 2)) for i in 1:N]
    return HMM(p, A, d)
end

function rand_gaussian_hmm_2d_light(N, D; sparse_trans=false)
    p = ones(N) / N
    A = sparse_trans ? sparse_trans_mat(N) : rand_trans_mat(N)
    d = [LightDiagNormal(randn(D), ones(D)) for i in 1:N]
    return HMM(p, A, d)
end

end
