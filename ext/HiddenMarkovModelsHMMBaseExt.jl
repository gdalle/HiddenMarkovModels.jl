module HiddenMarkovModelsHMMBaseExt

using HiddenMarkovModels: HiddenMarkovModels
using HiddenMarkovModels: StandardStateProcess, StandardObservationProcess
using HiddenMarkovModels: initial_distribution, transition_matrix, distributions
using HMMBase: HMMBase

function HiddenMarkovModels.HMM(hmm_base::HMMBase.HMM)
    p = copy(hmm_base.a)
    A = copy(hmm_base.A)
    distributions = copy.(hmm_base.B)
    state_process = StandardStateProcess(p, A)
    obs_process = StandardObservationProcess(distributions)
    return HiddenMarkovModels.HMM(state_process, obs_process)
end

function HMMBase.HMM(hmm::HiddenMarkovModels.HMM)
    a = initial_distribution(hmm.state_process)
    A = transition_matrix(hmm.state_process)
    B = distributions(hmm.obs_process)
    hmm_base = HMMBase.HMM(a, A, B)
    return hmm_base
end

end
