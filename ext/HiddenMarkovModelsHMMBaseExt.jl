module HiddenMarkovModelsHMMBaseExt

using HiddenMarkovModels: HiddenMarkovModels
using HMMBase: HMMBase

function HiddenMarkovModels.HMM(hmm_base::HMMBase.HMM)
    init = deepcopy(hmm_base.a)
    trans = deepcopy(hmm_base.A)
    dists = deepcopy(hmm_base.B)
    return HiddenMarkovModels.HMM(init, trans, dists)
end

function HMMBase.HMM(hmm::HiddenMarkovModels.HMM)
    a = deepcopy(hmm.init)
    A = deepcopy(hmm.trans)
    B = deepcopy(hmm.dists)
    return HMMBase.HMM(a, A, B)
end

function HiddenMarkovModels.initialization(hmm_base::HMMBase.HMM)
    return hmm_base.a
end

function HiddenMarkovModels.transition_matrix(hmm_base::HMMBase.HMM, ::Integer)
    return hmm_base.A
end

function HiddenMarkovModels.obs_distributions(hmm_base::HMMBase.HMM, ::Integer)
    return hmm_base.B
end

end
