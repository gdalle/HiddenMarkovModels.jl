module HiddenMarkovModelsChainRulesCoreExt

using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, rrule_via_ad
using DensityInterface: logdensityof
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using SimpleUnPack

function obs_logdensities_matrix(hmm::AbstractHMM, obs_seq::Vector)
    d = obs_distributions(hmm)
    logB = reduce(hcat, logdensityof.(d, (obs_seq[t],)) for t in 1:length(obs_seq))
    return logB
end

function _params_and_loglikelihoods(hmm::AbstractHMM, obs_seq::Vector)
    p = initialization(hmm)
    As = [transition_matrix(hmm, t) for t in 1:length(obs_seq)]
    logB = obs_logdensities_matrix(hmm, obs_seq)
    return p, As, logB
end

function ChainRulesCore.rrule(
    rc::RuleConfig, ::typeof(logdensityof), hmm::AbstractHMM, obs_seq::Vector
)
    (p, As, logB), pullback = rrule_via_ad(rc, _params_and_loglikelihoods, hmm, obs_seq)
    storage = HMMs.initialize_forward_backward(hmm, obs_seq)
    HMMs.forward_backward!(storage, hmm, obs_seq)
    @unpack logL, α, β, γ, c, B = storage
    T = length(obs_seq)

    function logdensityof_hmm_pullback(ΔlogL)
        Δp = ΔlogL .* Bβ[:, 1]
        ΔAs = [ΔlogL .* α[:, 1] .* B[:, 2]' .* β[:, 2]']
        @views for t in 2:(T - 1)
            push!(ΔAs, ΔlogL .* α[:, t] .* B[:, t + 1]' .* β[:, t + 1]')
        end
        ΔlogB = ΔlogL .* γ

        Δlogdensityof = NoTangent()
        _, Δhmm, Δobs_seq = pullback((Δp, ΔAs, ΔlogB))
        return Δlogdensityof, Δhmm, Δobs_seq
    end

    return logL[], logdensityof_hmm_pullback
end

end
