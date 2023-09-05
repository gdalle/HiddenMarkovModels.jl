module HiddenMarkovModelsChainRulesCoreExt

using ChainRulesCore:
    ChainRulesCore, NoTangent, ZeroTangent, RuleConfig, rrule_via_ad, @not_implemented
using DensityInterface: logdensityof
using HiddenMarkovModels
using SimpleUnPack

function _params_and_loglikelihoods(hmm::AbstractHMM, obs_seq)
    p = initial_distribution(hmm)
    A = transition_matrix(hmm)
    logB = HiddenMarkovModels.loglikelihoods(hmm, obs_seq)
    return p, A, logB
end

function ChainRulesCore.rrule(
    rc::RuleConfig, ::typeof(logdensityof), hmm::AbstractHMM, obs_seq
)
    error("Chain rule not yet fully implemented")
    (p, A, logB), pullback = rrule_via_ad(rc, _params_and_loglikelihoods, hmm, obs_seq)
    y = exp.(logB)
    fb = forward_backward(p, A, logB)
    logL = HiddenMarkovModels.loglikelihood(fb)
    @unpack α, β, γ, c, maxlogB = fb
    T = length(obs_seq)

    function logdensityof_hmm_pullback(ΔlogL)
        # Source: https://idiap.github.io/HMMGradients.jl/stable/1_intro/
        # TODO: adapt formulas with our logsumexp trick
        Δp = @not_implemented("todo")
        ΔA = @not_implemented("todo")
        ΔlogB = ΔlogL .* fb.γ

        Δlogdensityof = NoTangent()
        _, Δhmm, Δobs_seq = pullback((Δp, ΔA, ΔlogB))
        return Δlogdensityof, Δhmm, Δobs_seq
    end

    return logL, logdensityof_hmm_pullback
end

end
