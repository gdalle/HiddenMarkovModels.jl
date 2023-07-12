module HiddenMarkovModelsHMMBaseExt

using HiddenMarkovModels
using HMMBase: HMMBase

function HiddenMarkovModels.HMM(hmm_base::HMMBase.HMM)
    init = deepcopy(hmm_base.a)
    trans = deepcopy(hmm_base.A)
    dists = deepcopy(hmm_base.B)
    return HiddenMarkovModels.HMM(init, trans, dists)
end

function HMMBase.HMM(hmm::HiddenMarkovModels.HMM)
    a = deepcopy(initial_distribution(hmm))
    A = deepcopy(transition_matrix(hmm))
    B = deepcopy(obs_distribution.(Ref(hmm), 1:length(hmm)))
    return HMMBase.HMM(a, A, B)
end

end