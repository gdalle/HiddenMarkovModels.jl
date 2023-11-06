using BenchmarkTools
using Distributions
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using HiddenMarkovModels: LightDiagNormal
using SimpleUnPack
using Test

function test_allocations(hmm; T)
    obs_seq = rand(hmm, T).obs_seq

    ## Forward
    f = HMMs.initialize_forward(hmm, obs_seq)
    allocs = @ballocated HiddenMarkovModels.forward!($f, $hmm, $obs_seq) samples = 2
    @test allocs == 0

    ## Viterbi
    v = HMMs.initialize_viterbi(hmm, obs_seq)
    allocs = @ballocated HMMs.viterbi!($v, $hmm, $obs_seq) samples = 2
    @test allocs == 0

    ## Forward-backward
    fb = HMMs.initialize_forward_backward(hmm, obs_seq)
    allocs = @ballocated HMMs.forward_backward!($fb, $hmm, $obs_seq) samples = 2
    @test allocs == 0

    ## Baum-Welch
    fb = HMMs.initialize_forward_backward(hmm, obs_seq)
    R = eltype(hmm, obs_seq[1])
    logL_evolution = R[]
    sizehint!(logL_evolution, 2)
    allocs = @ballocated HMMs.baum_welch!(
        $fb,
        $logL_evolution,
        $hmm,
        $obs_seq;
        atol=-Inf,
        max_iterations=2,
        loglikelihood_increasing=false,
    ) samples = 2
    @test allocs == 0
end

N = 5
D = 3
T = 100

p = rand_prob_vec(N)
A = rand_trans_mat(N)
dists = [LightDiagNormal(randn(2), ones(2)) for i in 1:N]

hmm = HMM(p, A, dists)

test_allocations(hmm; T)
