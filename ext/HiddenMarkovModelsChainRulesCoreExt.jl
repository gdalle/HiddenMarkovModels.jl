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
    (p, A, logB), pullback = rrule_via_ad(rc, _params_and_loglikelihoods, hmm, obs_seq)
    fb = forward_backward(p, A, logB)
    logL = HiddenMarkovModels.loglikelihood(fb)
    @unpack α, β, γ, c, Bβscaled = fb
    T = length(obs_seq)

    function logdensityof_hmm_pullback(ΔlogL)
        Δp = ΔlogL .* Bβscaled[1]
        ΔA = ΔlogL .* α[1] .* Bβscaled[2]'
        for t in 2:(T - 1)
            ΔA .+= ΔlogL .* α[t] .* Bβscaled[t + 1]'
        end
        ΔlogB = ΔlogL .* reduce(hcat, γ)

        Δlogdensityof = NoTangent()
        _, Δhmm, Δobs_seq = pullback((Δp, ΔA, ΔlogB))
        return Δlogdensityof, Δhmm, Δobs_seq
    end

    return logL, logdensityof_hmm_pullback
end

end
