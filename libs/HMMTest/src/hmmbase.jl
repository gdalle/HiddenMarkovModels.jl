
function test_identical_hmmbase(
    rng::AbstractRNG,
    hmm::AbstractHMM,
    T::Integer;
    atol::Real=1e-5,
    hmm_guess::Union{Nothing,AbstractHMM}=nothing,
)
    @testset "HMMBase" begin
        sim = rand(rng, hmm, T)
        obs_mat = collect(reduce(hcat, sim.obs_seq)')

        obs_seq = vcat(sim.obs_seq, sim.obs_seq)
        seq_ends = [length(sim.obs_seq), 2 * length(sim.obs_seq)]

        hmm_base = HMMBase.HMM(deepcopy(hmm.init), deepcopy(hmm.trans), deepcopy(hmm.dists))

        logL_base = HMMBase.forward(hmm_base, obs_mat)[2]
        logL = logdensityof(hmm, obs_seq; seq_ends)
        @test sum(logL) ≈ 2logL_base

        α_base, logL_forward_base = HMMBase.forward(hmm_base, obs_mat)
        α, logL_forward = forward(hmm, obs_seq; seq_ends)
        @test isapprox(α[:, 1:T], α_base') && isapprox(α[:, (T + 1):(2T)], α_base')
        @test sum(logL_forward) ≈ 2logL_forward_base

        q_base = HMMBase.viterbi(hmm_base, obs_mat)
        q, logL_viterbi = viterbi(hmm, obs_seq; seq_ends)
        # Viterbi decoding can vary in case of (infrequent) ties
        @test mean(q[1:T] .== q_base) > 0.9 && mean(q[(T + 1):(2T)] .== q_base) > 0.9

        γ_base = HMMBase.posteriors(hmm_base, obs_mat)
        γ, logL_forward_backward = forward_backward(hmm, obs_seq; seq_ends)
        @test isapprox(γ[:, 1:T], γ_base') && isapprox(γ[:, (T + 1):(2T)], γ_base')

        if !isnothing(hmm_guess)
            hmm_guess_base = HMMBase.HMM(
                deepcopy(hmm_guess.init),
                deepcopy(hmm_guess.trans),
                deepcopy(hmm_guess.dists),
            )

            hmm_est_base, hist_base = HMMBase.fit_mle(
                hmm_guess_base, obs_mat; maxiter=10, tol=-Inf
            )
            logL_evolution_base = hist_base.logtots
            hmm_est, logL_evolution = baum_welch(
                hmm_guess, obs_seq; seq_ends, max_iterations=10, atol=-Inf
            )
            @test isapprox(
                logL_evolution[(begin + 1):end], 2 * logL_evolution_base[begin:(end - 1)]
            )
            hmm_est_base_converted = HMM(hmm_est_base.a, hmm_est_base.A, hmm_est_base.B)
            test_equal_hmms(hmm_est, hmm_est_base_converted, [nothing]; atol, init=true)
        end
    end
end
