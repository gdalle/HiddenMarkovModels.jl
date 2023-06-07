function forward_light!(αₜ, αₜ₊₁, b, p, A, op::ObservationProcess, obs_seq)
    T = length(obs_seq)
    likelihoods_vec!(b, op, obs_seq[1])
    αₜ .= p .* b
    c = inv(sum(αₜ))
    logL = -log(c)
    αₜ .*= c
    for t in 1:(T - 1)
        mul!(αₜ₊₁, A', αₜ)
        likelihoods_vec!(b, op, obs_seq[t + 1])
        αₜ₊₁ .*= b
        c = inv(sum(αₜ₊₁))
        logL -= log(c)
        αₜ₊₁ .*= c
        αₜ .= αₜ₊₁
        check_nan(αₜ)
    end
    return logL
end

"""
    DensityInterface.logdensityof(hmm, obs_seq)

Apply the forward algorithm to compute the loglikelihood of a sequence of observations.
"""
function DensityInterface.logdensityof(hmm::HMM, obs_seq)
    N = length(hmm)
    p = initial_distribution(hmm.state_process)
    A = transition_matrix(hmm.state_process)
    b = likelihoods_vec(hmm.obs_process, obs_seq[1])

    R = promote_type(eltype(p), eltype(A), eltype(b))
    αₜ = Vector{R}(undef, N)
    αₜ₊₁ = Vector{R}(undef, N)

    logL = forward_light!(αₜ, αₜ₊₁, b, p, A, hmm.obs_process, obs_seq)
    return logL
end
