
function test_type_stability(
    rng::AbstractRNG,
    hmm::AbstractHMM,
    control_seq::AbstractVector;
    seq_ends::AbstractVector{Int},
    hmm_guess::Union{Nothing,AbstractHMM}=nothing,
)
    @testset "Type stability" begin
        state_seq, obs_seq = rand(rng, hmm, control_seq)

        @test_opt target_modules = (HMMs,) rand(hmm, control_seq)
        @test_call target_modules = (HMMs,) rand(hmm, control_seq)

        @test_opt target_modules = (HMMs,) logdensityof(hmm, obs_seq, control_seq; seq_ends)
        @test_call target_modules = (HMMs,) logdensityof(
            hmm, obs_seq, control_seq; seq_ends
        )
        @test_opt target_modules = (HMMs,) logdensityof(
            hmm, obs_seq, state_seq; control_seq, seq_ends
        )
        @test_call target_modules = (HMMs,) logdensityof(
            hmm, obs_seq, state_seq; control_seq, seq_ends
        )

        @test_opt target_modules = (HMMs,) forward(hmm, obs_seq, control_seq; seq_ends)
        @test_call target_modules = (HMMs,) forward(hmm, obs_seq, control_seq; seq_ends)

        @test_opt target_modules = (HMMs,) viterbi(hmm, obs_seq, control_seq; seq_ends)
        @test_call target_modules = (HMMs,) viterbi(hmm, obs_seq, control_seq; seq_ends)

        @test_opt target_modules = (HMMs,) forward_backward(
            hmm, obs_seq, control_seq; seq_ends
        )
        @test_call target_modules = (HMMs,) forward_backward(
            hmm, obs_seq, control_seq; seq_ends
        )

        if !isnothing(hmm_guess)
            @test_opt target_modules = (HMMs,) baum_welch(
                hmm, obs_seq, control_seq; seq_ends, max_iterations=1
            )
            @test_call target_modules = (HMMs,) baum_welch(
                hmm, obs_seq, control_seq; seq_ends, max_iterations=1
            )
        end
    end
end
