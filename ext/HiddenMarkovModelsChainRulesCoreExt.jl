module HiddenMarkovModelsChainRulesCoreExt

using ChainRulesCore:
    ChainRulesCore, NoTangent, ZeroTangent, RuleConfig, rrule_via_ad, @not_implemented
using DensityInterface: logdensityof
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using SimpleUnPack

function obs_logdensities_matrix(hmm::AbstractHMM, obs_seq::Vector)
    d = obs_distributions(hmm)
    logB = reduce(hcat, logdensityof.(d, (obs_seq[t],)) for t in 1:length(obs_seq))
    return logB
end

function _params_and_loglikelihoods(hmm::AbstractHMM, obs_seq)
    p = initialization(hmm)
    A = transition_matrix(hmm)
    logB = obs_logdensities_matrix(hmm, obs_seq)
    return p, A, logB
end

function ChainRulesCore.rrule(
    rc::RuleConfig, ::typeof(logdensityof), hmm::AbstractHMM, obs_seq
)
    (p, A, logB), pullback = rrule_via_ad(rc, _params_and_loglikelihoods, hmm, obs_seq)
    fb = HMMs.initialize_forward_backward(hmm, obs_seq)
    HMMs.forward_backward!(fb, hmm, obs_seq)
    @unpack α, β, γ, c, Bβ = fb
    T = length(obs_seq)

    function logdensityof_hmm_pullback(ΔlogL)
        Δp = ΔlogL .* Bβ[:, 1]
        ΔA = ΔlogL .* α[:, 1] .* Bβ[:, 2]'
        @views for t in 2:(T - 1)
            ΔA .+= ΔlogL .* α[:, t] .* Bβ[:, t + 1]'
        end
        ΔlogB = ΔlogL .* γ

        Δlogdensityof = NoTangent()
        _, Δhmm, Δobs_seq = pullback((Δp, ΔA, ΔlogB))
        return Δlogdensityof, Δhmm, Δobs_seq
    end

    return fb.logL[], logdensityof_hmm_pullback
end

end
