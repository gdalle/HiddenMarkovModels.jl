function _params_and_loglikelihoods(
    hmm::AbstractHMM,
    obs_seq::Vector,
    control_seq::AbstractVecOrMat=Fill(nothing, duration(obs_seq));
    seq_ends::AbstractVector{Int}=Fill(duration(obs_seq), 1),
)
    init = initialization(hmm)
    trans_by_time = mapreduce(dcat, 1:duration(obs_seq)) do t
        transition_matrix(hmm, control_seq, t)
    end
    logB = mapreduce(hcat, 1:duration(obs_seq)) do t
        logdensityof.(obs_distributions(hmm, control_seq, t), (at_time(obs_seq, t),))
    end
    return init, trans_by_time, logB
end

function ChainRulesCore.rrule(
    rc::RuleConfig,
    ::typeof(logdensityof),
    hmm::AbstractHMM,
    obs_seq::AbstractVecOrMat,
    control_seq::AbstractVecOrMat=Fill(nothing, duration(obs_seq));
    seq_ends::AbstractVector{Int}=Fill(duration(obs_seq), 1),
)
    _, pullback = rrule_via_ad(
        rc, _params_and_loglikelihoods, hmm, obs_seq, control_seq; seq_ends
    )
    fb_storage = initialize_forward_backward(hmm, obs_seq, control_seq; seq_ends)
    forward_backward!(fb_storage, hmm, obs_seq, control_seq; seq_ends)
    (; logL, α, γ, Bβ) = fb_storage
    N, T = length(hmm), duration(obs_seq)
    R = eltype(α)

    Δinit = zeros(R, N)
    Δtrans_by_time = zeros(R, N, N, T)
    @views for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        Δinit .+= Bβ[:, t1]
        for t in t1:(t2 - 1)
            Δtrans_by_time[:, :, t] .= α[:, t] .* Bβ[:, t + 1]'
        end
    end
    ΔlogB = γ

    function logdensityof_hmm_pullback(ΔlogL)
        _, Δhmm, Δobs_seq, Δcontrol_seq = pullback((
            ΔlogL .* Δinit, ΔlogL .* Δtrans_by_time, ΔlogL .* ΔlogB
        ))
        Δlogdensityof = NoTangent()
        return Δlogdensityof, Δhmm, Δobs_seq, Δcontrol_seq
    end

    return sum(logL), logdensityof_hmm_pullback
end
