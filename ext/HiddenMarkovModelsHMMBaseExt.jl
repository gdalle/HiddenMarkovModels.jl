module HiddenMarkovModelsHMMBaseExt

using HiddenMarkovModels: HiddenMarkovModels
using HiddenMarkovModels: StandardTransitions, VectorEmissions
using HiddenMarkovModels: initial_distribution, transition_matrix, emission_distributions
using HMMBase: HMMBase

function HiddenMarkovModels.HMM(hmm_base::HMMBase.HMM)
    p = copy(hmm_base.a)
    A = copy(hmm_base.A)
    distributions = copy.(hmm_base.B)
    transitions = StandardTransitions(p, A)
    emissions = VectorEmissions(distributions)
    return HiddenMarkovModels.HMM(transitions, emissions)
end

function HMMBase.HMM(hmm::HiddenMarkovModels.HMM)
    a = initial_distribution(hmm)
    A = transition_matrix(hmm)
    B = emission_distributions(hmm)
    hmm_base = HMMBase.HMM(a, A, B)
    return hmm_base
end

end
