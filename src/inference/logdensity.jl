function forward_light!(αₜ, αₜ₊₁, logb, p, A, op::ObservationProcess, obs_seq)
    T = length(obs_seq)
    loglikelihoods_vec!(logb, op, obs_seq[1])
    m = maximum(logb)
    αₜ .= p .* exp.(logb .- m)
    c = inv(sum(αₜ))
    αₜ .*= c
    logL = -log(c) + m
    for t in 1:(T - 1)
        loglikelihoods_vec!(logb, op, obs_seq[t + 1])
        m = maximum(logb)
        mul!(αₜ₊₁, A', αₜ)
        αₜ₊₁ .*= exp.(logb .- m)
        c = inv(sum(αₜ₊₁))
        αₜ₊₁ .*= c
        αₜ .= αₜ₊₁
        logL += -log(c) + m
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
    logb = loglikelihoods_vec(hmm.obs_process, obs_seq[1])

    R = promote_type(eltype(p), eltype(A), eltype(logb))
    αₜ = Vector{R}(undef, N)
    αₜ₊₁ = Vector{R}(undef, N)
    logL = forward_light!(αₜ, αₜ₊₁, logb, p, A, hmm.obs_process, obs_seq)
    return logL
end
