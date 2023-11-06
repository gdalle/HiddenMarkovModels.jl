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
    allocs = @ballocated HiddenMarkovModels.forward!($f, $hmm, $obs_seq)
    @test allocs == 0

    ## Viterbi
    v = HMMs.initialize_viterbi(hmm, obs_seq)
    allocs = @ballocated HMMs.viterbi!($v, $hmm, $obs_seq)
    @test allocs == 0

    ## Forward-backward
    fb = HMMs.initialize_forward_backward(hmm, obs_seq)
    allocs = @ballocated HMMs.forward_backward!($fb, $hmm, $obs_seq)
    @test allocs == 0

    ## Baum-Welch
    nb_seqs = 2
    obs_seqs = [obs_seq for _ in 1:nb_seqs]
    bw = HMMs.initialize_baum_welch(hmm, obs_seqs, nb_seqs; max_iterations=2)
    allocs = @ballocated HMMs.baum_welch!(
        $hmm,
        $bw,
        $obs_seqs;
        atol=-Inf,
        max_iterations=2,
        check_loglikelihood_increasing=false,
    )
    @test_broken allocs == 0  # @threads introduces type instability, see https://discourse.julialang.org/t/type-instability-because-of-threads-boxing-variables/78395/
end

N = 5
D = 3
T = 100

p = rand_prob_vec(N)
A = rand_trans_mat(N)
dists = [LightDiagNormal(randn(2), ones(2)) for i in 1:N]

hmm = HMM(p, A, dists)
obs_seq = rand(hmm, T).obs_seq

test_allocations(hmm; T)
