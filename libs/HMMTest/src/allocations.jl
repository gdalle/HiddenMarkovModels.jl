
function test_allocations(
    rng::AbstractRNG,
    hmm::AbstractHMM,
    hmm_guess::Union{Nothing,AbstractHMM}=nothing;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
)
    @testset "Allocations" begin
        obs_seq = mapreduce(vcat, eachindex(seq_ends)) do k
            t1, t2 = seq_limits(seq_ends, k)
            rand(rng, hmm, control_seq[t1:t2]).obs_seq
        end

        t1, t2 = 1, seq_ends[1]

        ## Forward
        f_storage = HMMs.initialize_forward(hmm, obs_seq, control_seq; seq_ends)
        allocs = @ballocated HMMs.forward!(
            $f_storage, $hmm, $obs_seq, $control_seq, $t1, $t2
        ) evals = 1 samples = 1
        @test allocs == 0

        ## Viterbi
        v_storage = HMMs.initialize_viterbi(hmm, obs_seq, control_seq; seq_ends)
        allocs = @ballocated HMMs.viterbi!(
            $v_storage, $hmm, $obs_seq, $control_seq, $t1, $t2
        ) evals = 1 samples = 1
        @test allocs == 0

        ## Forward-backward
        fb_storage = HMMs.initialize_forward_backward(hmm, obs_seq, control_seq; seq_ends)
        allocs = @ballocated HMMs.forward_backward!(
            $fb_storage, $hmm, $obs_seq, $control_seq, $t1, $t2
        ) evals = 1 samples = 1
        @test allocs == 0

        ## Baum-Welch
        if !isnothing(hmm_guess)
            fb_storage = HMMs.initialize_forward_backward(
                hmm_guess, obs_seq, control_seq; seq_ends
            )
            HMMs.forward_backward!(fb_storage, hmm, obs_seq, control_seq; seq_ends)
            allocs = @ballocated fit!(
                $hmm_guess, $fb_storage, $obs_seq, $control_seq; seq_ends=$seq_ends
            ) evals = 1 samples = 1 setup = (hmm_guess = deepcopy($hmm))
            @test allocs == 0
        end
    end
end
