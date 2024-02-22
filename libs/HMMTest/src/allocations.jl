
function test_allocations(
    rng::AbstractRNG,
    hmm::AbstractHMM,
    control_seq::AbstractVector;
    seq_ends::AbstractVector{Int},
    hmm_guess::Union{Nothing,AbstractHMM}=nothing,
)
    @testset "Allocations" begin
        obs_seq = mapreduce(vcat, eachindex(seq_ends)) do k
            t1, t2 = seq_limits(seq_ends, k)
            rand(rng, hmm, control_seq[t1:t2]).obs_seq
        end

        t1, t2 = 1, seq_ends[1]

        ## Forward

        f_storage = HMMs.initialize_forward(hmm, obs_seq, control_seq; seq_ends)
        allocs_f = @ballocated HMMs.forward!(
            $f_storage, $hmm, $obs_seq, $control_seq, $t1, $t2
        ) evals = 1 samples = 1
        @test allocs_f == 0

        ## Viterbi

        v_storage = HMMs.initialize_viterbi(hmm, obs_seq, control_seq; seq_ends)
        allocs_v = @ballocated HMMs.viterbi!(
            $v_storage, $hmm, $obs_seq, $control_seq, $t1, $t2
        ) evals = 1 samples = 1
        @test allocs_v == 0

        ## Forward-backward

        fb_storage = HMMs.initialize_forward_backward(hmm, obs_seq, control_seq; seq_ends)
        allocs_fb = @ballocated HMMs.forward_backward!(
            $fb_storage, $hmm, $obs_seq, $control_seq, $t1, $t2
        ) evals = 1 samples = 1
        @test allocs_fb == 0

        ## Baum-Welch

        if !isnothing(hmm_guess)
            fb_storage = HMMs.initialize_forward_backward(
                hmm_guess, obs_seq, control_seq; seq_ends
            )
            HMMs.forward_backward!(fb_storage, hmm, obs_seq, control_seq; seq_ends)
            allocs_bw = @ballocated fit!(
                hmm_guess_copy, $fb_storage, $obs_seq, $control_seq; seq_ends=$seq_ends
            ) evals = 1 samples = 1 setup = (hmm_guess_copy = deepcopy($hmm_guess))
            @test_broken allocs_bw == 0
        end
    end
end
