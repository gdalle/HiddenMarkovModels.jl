infnorm(x) = maximum(abs, x)

function test_equal_hmms(
    hmm1::AbstractHMM,
    hmm2::AbstractHMM,
    control_seq::AbstractVector;
    atol::Real,
    init::Bool,
    flip::Bool=false,
)
    if init
        init1 = initialization(hmm1)
        init2 = initialization(hmm2)
        if flip
            @test !isapprox(init1, init2; atol, norm=infnorm)
        else
            @test isapprox(init1, init2; atol, norm=infnorm)
        end
    end

    for control in control_seq
        trans1 = transition_matrix(hmm1, control)
        trans2 = transition_matrix(hmm2, control)
        if flip
            @test !isapprox(trans1, trans2; atol, norm=infnorm)
        else
            @test isapprox(trans1, trans2; atol, norm=infnorm)
        end
    end

    for control in control_seq
        dists1 = obs_distributions(hmm1, control)
        dists2 = obs_distributions(hmm2, control)
        for (dist1, dist2) in zip(dists1, dists2)
            for field in fieldnames(typeof(dist1))
                if startswith(string(field), "log") ||
                    contains("σ", string(field)) ||
                    contains("Σ", string(field))
                    continue
                end
                x1 = getfield(dist1, field)
                x2 = getfield(dist2, field)
                if flip
                    @test !isapprox(x1, x2; atol, norm=infnorm)
                else
                    @test isapprox(x1, x2; atol, norm=infnorm)
                end
            end
        end
    end

    return nothing
end

function test_coherent_algorithms(
    rng::AbstractRNG,
    hmm::AbstractHMM,
    control_seq::AbstractVector;
    seq_ends::AbstractVector{Int},
    hmm_guess::Union{Nothing,AbstractHMM}=nothing,
    atol::Real=0.05,
    init::Bool=true,
)
    @testset "Coherence" begin
        simulations = map(eachindex(seq_ends)) do k
            t1, t2 = seq_limits(seq_ends, k)
            rand(rng, hmm, control_seq[t1:t2])
        end

        state_seqs = [sim.state_seq for sim in simulations]
        obs_seqs = [sim.obs_seq for sim in simulations]

        state_seq = reduce(vcat, state_seqs)
        obs_seq = reduce(vcat, obs_seqs)

        logL = logdensityof(hmm, obs_seq, control_seq; seq_ends)
        logL_joint = joint_logdensityof(hmm, obs_seq, state_seq, control_seq; seq_ends)

        q, logL_viterbi = viterbi(hmm, obs_seq, control_seq; seq_ends)
        @test sum(logL_viterbi) > logL_joint
        @test sum(logL_viterbi) ≈ joint_logdensityof(hmm, obs_seq, q, control_seq; seq_ends)

        α, logL_forward = forward(hmm, obs_seq, control_seq; seq_ends)
        @test sum(logL_forward) .≈ logL

        γ, logL_forward_backward = forward_backward(hmm, obs_seq, control_seq; seq_ends)
        @test sum(logL_forward_backward) ≈ logL
        @test all(α[:, seq_ends[k]] ≈ γ[:, seq_ends[k]] for k in eachindex(seq_ends))

        if !isnothing(hmm_guess)
            hmm_est, logL_evolution = baum_welch(hmm_guess, obs_seq, control_seq; seq_ends)
            @test all(>=(0), diff(logL_evolution))
            test_equal_hmms(hmm, hmm_guess, control_seq[1:2]; atol, init, flip=true)
            test_equal_hmms(hmm, hmm_est, control_seq[1:2]; atol, init)
        end
    end
end
