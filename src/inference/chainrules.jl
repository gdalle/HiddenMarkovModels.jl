function _params_and_loglikelihoods(
    hmm::AbstractHMM, obs_seq::Vector, control_seq::AbstractVector=no_controls(obs_seq)
)
    T = length(eachindex(obs_seq, controls_seq))
    init = initialization(hmm)
    trans_by_time = [transition_matrix(hmm, control_seq[t]) for t in 1:T]
    logB_columns = map(1:T) do t
        logdensityof.(obs_distributions(hmm, control_seq[t]), (obs_seq[t],))
    end
    logB = reduce(hcat, logB_columns)
    return init, trans_by_time, logB
end

function ChainRulesCore.rrule(
    rc::RuleConfig,
    ::typeof(logdensityof),
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=no_controls(obs_seq),
)
    _, pullback = rrule_via_ad(rc, _params_and_loglikelihoods, hmm, obs_seq, control_seq)
    fb_storage = initialize_forward_backward(hmm, obs_seq, control_seq)
    forward_backward!(fb_storage, hmm, obs_seq, control_seq)
    @unpack logL, α, β, γ, c, B = fb_storage
    T = length(eachindex(obs_seq, controls_seq))
    R = eltype(α)

    function logdensityof_hmm_pullback(ΔlogL)
        Δinit = ΔlogL .* B[:, 1] .* β[:, 1]
        Δtrans_by_time = Vector{Matrix{R}}(undef, T - 1)
        @views for t in 1:(T - 1)
            if t == 1
                Δtrans_by_time[t] = ΔlogL .* α[:, 1] .* B[:, 2]' .* β[:, 2]'
            else
                Δtrans_by_time[t] = ΔlogL .* α[:, t] .* B[:, t + 1]' .* β[:, t + 1]'
            end
        end
        ΔlogB = ΔlogL .* γ

        Δlogdensityof = NoTangent()
        _, Δhmm, Δobs_seq, Δcontrol_seq = pullback((Δinit, Δtrans_by_time, ΔlogB))
        return Δlogdensityof, Δhmm, Δobs_seq, Δcontrol_seq
    end

    return logL[], logdensityof_hmm_pullback
end