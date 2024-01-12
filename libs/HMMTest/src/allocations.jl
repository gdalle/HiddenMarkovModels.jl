
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

        ## Forward
        forward(hmm, obs_seq, control_seq; seq_ends)  # compile
        f_storage = HMMs.initialize_forward(hmm, obs_seq, control_seq; seq_ends)
        allocs = @allocated HMMs.forward!(f_storage, hmm, obs_seq, control_seq; seq_ends)
        @test allocs == 0

        ## Viterbi
        viterbi(hmm, obs_seq, control_seq; seq_ends)  # compile
        v_storage = HMMs.initialize_viterbi(hmm, obs_seq, control_seq; seq_ends)
        allocs = @allocated HMMs.viterbi!(v_storage, hmm, obs_seq, control_seq; seq_ends)
        @test allocs == 0

        ## Forward-backward
        forward_backward(hmm, obs_seq, control_seq; seq_ends)  # compile
        fb_storage = HMMs.initialize_forward_backward(hmm, obs_seq, control_seq; seq_ends)
        allocs = @allocated HMMs.forward_backward!(
            fb_storage, hmm, obs_seq, control_seq; seq_ends
        )
        @test allocs == 0

        if !isnothing(hmm_guess)
            ## Baum-Welch
            baum_welch(hmm_guess, obs_seq, control_seq; seq_ends, max_iterations=1)  # compile
            fb_storage = HMMs.initialize_forward_backward(
                hmm_guess, obs_seq, control_seq; seq_ends
            )
            logL_evolution = Float64[]
            sizehint!(logL_evolution, 1)
            hmm_guess = deepcopy(hmm_guess)
            allocs = @allocated HMMs.baum_welch!(
                fb_storage,
                logL_evolution,
                hmm_guess,
                obs_seq,
                control_seq;
                seq_ends,
                atol=-Inf,
                max_iterations=1,
                loglikelihood_increasing=false,
            )
            @test allocs == 0
        end
    end
end
