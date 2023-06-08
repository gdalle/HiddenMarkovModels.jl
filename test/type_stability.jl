using Distributions: DiagNormal, Normal, PDiagMat
using HiddenMarkovModels
using JET
using Test

function test_type_stability(hmm; T)
    (; state_seq, obs_seq) = rand(hmm, T)

    @testset "Logdensity" begin
        @inferred logdensityof(hmm, obs_seq)
        @test_opt target_modules = (HiddenMarkovModels,) logdensityof(hmm, obs_seq)
        @test_call target_modules = (HiddenMarkovModels,) logdensityof(hmm, obs_seq)
    end

    @testset "Viterbi" begin
        @inferred viterbi(hmm, obs_seq)
        @test_opt target_modules = (HiddenMarkovModels,) viterbi(hmm, obs_seq)
        @test_call target_modules = (HiddenMarkovModels,) viterbi(hmm, obs_seq)
    end

    @testset "Forward-backward" begin
        @inferred forward_backward(hmm, obs_seq)
        @test_opt target_modules = (HiddenMarkovModels,) forward_backward(hmm, obs_seq)
        @test_call target_modules = (HiddenMarkovModels,) forward_backward(hmm, obs_seq)
    end

    @testset "Baum-Welch" begin
        @inferred baum_welch(hmm, [obs_seq])
        @test_opt target_modules = (HiddenMarkovModels,) baum_welch(hmm, [obs_seq])
        @test_call target_modules = (HiddenMarkovModels,) baum_welch(hmm, [obs_seq])
    end
end

N = 5
D = 3

sp = StandardStateProcess(rand_prob_vec(N), rand_trans_mat(N))
op = StandardObservationProcess([Normal(randn(), 1.0) for i in 1:N])
hmm = HMM(sp, op)

test_type_stability(hmm; T=100)

op_multi = StandardObservationProcess([
    DiagNormal(randn(D), PDiagMat(ones(D))) for n in 1:N
]);
hmm_multi = HMM(sp, op_multi)

test_type_stability(hmm_multi; T=100)
