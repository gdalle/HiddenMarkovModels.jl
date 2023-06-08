using Distributions: Normal
using HiddenMarkovModels
using JET
using Test

function test_type_stability(hmm; T)
    (; state_seq, obs_seq) = rand(hmm, T)

    @testset "Logdensity" begin
        for scale in (NormalScale(), LogScale())
            @inferred logdensityof(hmm, obs_seq, scale)
            @test_opt target_modules = (HiddenMarkovModels,) logdensityof(
                hmm, obs_seq, scale
            )
            @test_call target_modules = (HiddenMarkovModels,) logdensityof(
                hmm, obs_seq, scale
            )
        end
    end

    @testset "Viterbi" begin
        for scale in (NormalScale(), LogScale())
            @inferred viterbi(hmm, obs_seq, scale)
            @test_opt target_modules = (HiddenMarkovModels,) viterbi(hmm, obs_seq, scale)
            @test_call target_modules = (HiddenMarkovModels,) viterbi(hmm, obs_seq, scale)
        end
    end

    @testset "Forward-backward" begin
        for scale in (NormalScale(), LogScale())
            @inferred forward_backward(hmm, obs_seq, scale)
            @test_opt target_modules = (HiddenMarkovModels,) forward_backward(
                hmm, obs_seq, scale
            )
            @test_call target_modules = (HiddenMarkovModels,) forward_backward(
                hmm, obs_seq, scale
            )
        end
    end

    @testset "Baum-Welch" begin
        for scale in (NormalScale(), LogScale())
            @inferred baum_welch(hmm, [obs_seq], scale)
            @test_opt target_modules = (HiddenMarkovModels,) baum_welch(
                hmm, [obs_seq], scale
            )
            @test_call target_modules = (HiddenMarkovModels,) baum_welch(
                hmm, [obs_seq], scale
            )
        end
    end
end

N = 5
sp = StandardStateProcess(rand_prob_vec(N), rand_trans_mat(N))
op = StandardObservationProcess([Normal(randn(), 1.0) for i in 1:N])
hmm = HMM(sp, op)

test_type_stability(hmm; T=100)
