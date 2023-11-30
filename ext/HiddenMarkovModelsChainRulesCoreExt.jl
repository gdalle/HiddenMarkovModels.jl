module HiddenMarkovModelsChainRulesCoreExt

using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, rrule_via_ad
using DensityInterface: logdensityof
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using SimpleUnPack

function obs_logdensities_matrix(hmm::AbstractHMM, obs_seq::Vector)
    logB = reduce(
        hcat,
        logdensityof.(obs_distributions(hmm, t), (obs_seq[t],)) for t in 1:length(obs_seq)
    )
    return logB
end

function _params_and_loglikelihoods(hmm::AbstractHMM, obs_seq::Vector)
    init = initialization(hmm)
    trans_by_time = [transition_matrix(hmm, t) for t in 1:length(obs_seq)]
    logB = obs_logdensities_matrix(hmm, obs_seq)
    return init, trans_by_time, logB
end

function ChainRulesCore.rrule(
    rc::RuleConfig, ::typeof(logdensityof), hmm::AbstractHMM, obs_seq::Vector
)
    _, pullback = rrule_via_ad(rc, _params_and_loglikelihoods, hmm, obs_seq)
    storage = HMMs.initialize_forward_backward(hmm, obs_seq)
    logL = HMMs.forward_backward!(storage, hmm, obs_seq)
    @unpack α, β, γ, c, B = storage
    T = length(obs_seq)

    function logdensityof_hmm_pullback(ΔlogL)
        Δinit = ΔlogL .* B[:, 1] .* β[:, 1]
        Δtrans_by_time = [ΔlogL .* α[:, 1] .* B[:, 2]' .* β[:, 2]']
        @views for t in 2:(T - 1)
            push!(Δtrans_by_time, ΔlogL .* α[:, t] .* B[:, t + 1]' .* β[:, t + 1]')
        end
        ΔlogB = ΔlogL .* γ

        Δlogdensityof = NoTangent()
        _, Δhmm, Δobs_seq = pullback((Δinit, Δtrans_by_time, ΔlogB))
        return Δlogdensityof, Δhmm, Δobs_seq
    end

    return logL, logdensityof_hmm_pullback
end

end
