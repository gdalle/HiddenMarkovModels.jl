"""
    forward_backward(hmm, obs_seq, scale=LogScale())

Apply the forward-backward algorithm to estimate the posterior state marginals of an HMM.
"""
function forward_backward(hmm::HMM, obs_seq, scale::Scale=LogScale())
    B = likelihoods(hmm.obs_process, obs_seq, scale)
    fb = initialize_forward_backward(hmm.state_process, B, scale)
    forward_backward!(fb, hmm.state_process, B)
    return fb
end

function loglikelihood(fbs::Vector{<:AbstractForwardBackwardStorage{R}}) where {R}
    logL = zero(R)
    for fb in fbs
        logL += loglikelihood(fb)
    end
    return logL
end
