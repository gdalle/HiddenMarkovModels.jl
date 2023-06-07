function forward_light!(αₜ, αₜ₊₁, b, p, A, op::ObservationProcess, obs_seq)
    T = length(obs_seq)
    likelihoods_vec!(b, op, obs_seq[1])
    αₜ .= p .* b
    c = inv(sum(αₜ))
    logL = -log(c)
    αₜ .*= c
    for t in 1:(T - 1)
        likelihoods_vec!(b, op, obs_seq[t + 1])
        mul!(αₜ₊₁, A', αₜ)
        αₜ₊₁ .*= b
        c = inv(sum(αₜ₊₁))
        logL -= log(c)
        αₜ₊₁ .*= c
        αₜ .= αₜ₊₁
    end
    return logL
end

function forward_light_log!(
    logαₜ, logαₜA, logb, logp, logA, op::ObservationProcess, obs_seq
)
    N, T = length(op), length(obs_seq)
    loglikelihoods_vec!(logb, op, obs_seq[1])
    logαₜ .= logp .+ logb
    for t in 1:(T - 1)
        loglikelihoods_vec!(logb, op, obs_seq[t + 1])
        logαₜA .= logαₜ .+ logA
        for j in 1:N
            logαₜ[j] = logsumexp(logαₜA[:, j])
        end
        logαₜ .+= logb
    end
    logL = logsumexp(logαₜ)
    return logL
end

"""
    DensityInterface.logdensityof(hmm, obs_seq, scale=LogScale())

Apply the forward algorithm to compute the loglikelihood of a sequence of observations.
"""
function DensityInterface.logdensityof(hmm::HMM, obs_seq)
    return logdensityof(hmm, obs_seq, LogScale())
end

function DensityInterface.logdensityof(hmm::HMM, obs_seq, ::NormalScale)
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

function DensityInterface.logdensityof(hmm::HMM, obs_seq, ::LogScale)
    N = length(hmm)
    logp = log_initial_distribution(hmm.state_process)
    logA = log_transition_matrix(hmm.state_process)
    logb = loglikelihoods_vec(hmm.obs_process, obs_seq[1])

    R = promote_type(eltype(logp), eltype(logA), eltype(logb))
    logαₜ = Vector{R}(undef, N)
    logαₜA = Matrix{R}(undef, N, N)
    logL = forward_light_log!(logαₜ, logαₜA, logb, logp, logA, hmm.obs_process, obs_seq)
    return logL
end
