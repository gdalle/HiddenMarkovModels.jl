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
    f = HiddenMarkovModels.initialize_forward(hmm, obs_seq)
    allocs = @ballocated HiddenMarkovModels.forward!($f, $hmm, $obs_seq)
    @test allocs == 0

    ## Viterbi
    v = HiddenMarkovModels.initialize_viterbi(hmm, obs_seq)
    allocs = @ballocated HiddenMarkovModels.viterbi!($v, $hmm, $obs_seq)
    @test allocs == 0

    ## Forward-backward
    fb = HiddenMarkovModels.initialize_forward_backward(hmm, obs_seq)
    allocs = @ballocated HiddenMarkovModels.forward_backward!($fb, $hmm, $obs_seq)
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
