using BenchmarkTools
using Distributions
using Distributions: PDiagMat
using HiddenMarkovModels
using SimpleUnPack
using Test

function test_allocations(hmm; T)
    p = initialization(hmm)
    A = transition_matrix(hmm)
    @unpack state_seq, obs_seq = rand(hmm, T)

    ## Forward
    logb = HMMs.loglikelihoods_vec(hmm, obs_seq[1])
    αₜ = zeros(N)
    αₜ₊₁ = zeros(N)
    allocs = @ballocated HMMs.forward!($αₜ, $αₜ₊₁, $logb, $p, $A, $hmm, $obs_seq)
    @test allocs == 0

    ## Viterbi
    logb = HMMs.loglikelihoods_vec(hmm, obs_seq[1])
    δₜ = zeros(N)
    δₜ₋₁ = zeros(N)
    δA_tmp = zeros(N)
    ψ = zeros(Int, N, T)
    q = zeros(Int, T)
    allocs = @ballocated HMMs.viterbi!(
        $q, $δₜ, $δₜ₋₁, $δA_tmp, $ψ, $logb, $p, $A, $hmm, $obs_seq
    )
    @test allocs == 0

    ## Forward-backward
    logB = HMMs.loglikelihoods(hmm, obs_seq)
    fb = HMMs.initialize_forward_backward(p, A, logB)
    allocs = @ballocated HMMs.forward_backward!($fb, $p, $A, $logB)
    @test allocs == 0
end

N = 5
D = 3
T = 100

p = rand_prob_vec(N)
A = rand_trans_mat(N)
dists = [Normal(randn(), 1.0) for i in 1:N]

hmm = HMM(p, A, dists)

test_allocations(hmm; T)
