using BenchmarkTools
using HiddenMarkovModels
using HiddenMarkovModels.HMMTest
import HiddenMarkovModels as HMMs
using Test

function test_allocations(hmm; T)
    obs_seq = rand(hmm, T).obs_seq
    nb_seqs = 2
    obs_seqs = [rand(hmm, T).obs_seq for _ in 1:nb_seqs]

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
    fbs = [HMMs.initialize_forward_backward(hmm, obs_seqs[k]) for k in eachindex(obs_seqs)]
    bw = HMMs.initialize_baum_welch(hmm, obs_seqs, nb_seqs)
    obs_seqs_concat = reduce(vcat, obs_seqs)
    HMMs.forward_backward!(fbs, hmm, obs_seqs, nb_seqs)
    HMMs.update_sufficient_statistics!(bw, fbs)
    fit!(hmm, bw, obs_seqs_concat)
    allocs = @ballocated HMMs.update_sufficient_statistics!($bw, $fbs)
    @test allocs == 0
    allocs = @ballocated fit!($hmm, $bw, $obs_seqs_concat)
    @test allocs == 0
end

N, D, T = 3, 2, 100

@testset "Normal" begin
    test_allocations(rand_gaussian_hmm_1d(N); T)
end

@testset "Normal sparse" begin
    # see https://discourse.julialang.org/t/why-does-mul-u-a-v-allocate-when-a-is-sparse-and-u-v-are-views/105995
    @test_skip test_allocations(rand_gaussian_hmm_1d(N; sparse_trans=true); T)
end

@testset "LightDiagNormal" begin
    test_allocations(rand_gaussian_hmm_2d_light(N, D); T)
end
