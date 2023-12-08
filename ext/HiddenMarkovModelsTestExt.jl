module HiddenMarkovModelsTestExt

using HiddenMarkovModels
using HiddenMarkovModels: seq_limits, test_equal_hmms
using Random: AbstractRNG
using Test

infnorm(x) = maximum(abs, x)

function check_equal_hmms(
    hmm1::AbstractHMM,
    hmm2::AbstractHMM;
    control_seq=[nothing],
    atol::Real=0.1,
    init::Bool=true,
    test::Bool=true,
)
    equal_check = true

    if init
        init1 = initialization(hmm1)
        init2 = initialization(hmm2)
        test && @test isapprox(init1, init2; atol, norm=infnorm)
        equal_check = equal_check && isapprox(init1, init2; atol, norm=infnorm)
    end

    for control in control_seq
        trans1 = transition_matrix(hmm1, control)
        trans2 = transition_matrix(hmm2, control)
        test && @test isapprox(trans1, trans2; atol, norm=infnorm)
        equal_check = equal_check && isapprox(trans1, trans2; atol, norm=infnorm)
    end

    for control in control_seq
        dists1 = obs_distributions(hmm1, control)
        dists2 = obs_distributions(hmm2, control)
        for (dist1, dist2) in zip(dists1, dists2)
            for field in fieldnames(typeof(dist1))
                if startswith(string(field), "log")
                    continue
                end
                x1 = getfield(dist1, field)
                x2 = getfield(dist2, field)
                test && @test isapprox(x1, x2; atol, norm=infnorm)
                equal_check = equal_check && isapprox(x1, x2; atol, norm=infnorm)
            end
        end
    end

    return equal_check
end

function HiddenMarkovModels.test_equal_hmms(
    hmm1::AbstractHMM,
    hmm2::AbstractHMM;
    control_seq=[nothing],
    atol::Real=0.1,
    init::Bool=true,
)
    check_equal_hmms(hmm1, hmm2; control_seq, atol, init, test=true)
    return nothing
end

function HiddenMarkovModels.test_coherent_algorithms(
    rng::AbstractRNG,
    hmm::AbstractHMM,
    hmm_guess::Union{Nothing,AbstractHMM}=nothing;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    atol::Real=0.1,
    init::Bool=false,
)
    simulations = map(eachindex(seq_ends)) do k
        t1, t2 = seq_limits(seq_ends, k)
        rand(rng, hmm, control_seq[t1:t2])
    end

    state_seqs = [sim.state_seq for sim in simulations]
    obs_seqs = [sim.obs_seq for sim in simulations]

    state_seq = reduce(vcat, state_seqs)
    obs_seq = reduce(vcat, obs_seqs)

    logL = logdensityof(hmm, obs_seq; control_seq, seq_ends)
    logL_joint = logdensityof(hmm, obs_seq, state_seq; control_seq, seq_ends)

    q, logL_viterbi = viterbi(hmm, obs_seq; control_seq, seq_ends)
    @test logL_viterbi > logL_joint
    @test logL_viterbi ≈ logdensityof(hmm, obs_seq, q; control_seq, seq_ends)

    α, logL_forward = forward(hmm, obs_seq; control_seq, seq_ends)
    @test logL_forward ≈ logL

    γ, logL_forward_backward = forward_backward(hmm, obs_seq; control_seq, seq_ends)
    @test logL_forward_backward ≈ logL
    @test all(α[:, seq_ends[k]] ≈ γ[:, seq_ends[k]] for k in eachindex(seq_ends))

    if !isnothing(hmm_guess)
        hmm_est, logL_evolution = baum_welch(hmm_guess, obs_seq; control_seq, seq_ends)
        @test all(>=(0), diff(logL_evolution))
        @test !check_equal_hmms(
            hmm, hmm_guess; control_seq=control_seq[1:2], atol, test=false
        )
        test_equal_hmms(hmm, hmm_est; control_seq=control_seq[1:2], atol, init)
    end

    return nothing
end

end
