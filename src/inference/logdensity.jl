function forward_light!(αₜ, αₜ₊₁, b, p, A, op::ObservationProcess, obs_seq)
    T = size(α, 2)
    likelihoods_vec!(b, op, obs_seq[1])
    αₜ .= p .* b
    c = inv(sum(αₜ))
    logL = -log(c)
    α_vec .*= c
    for t in 1:(T - 1)
        mul!(αₜ₊₁, A', αₜ)
        likelihoods_vec!(b, op, obs_seq[t + 1])
        αₜ₊₁ .*= b
        c = inv(sum(αₜ₊₁))
        logL -= log(c)
        αₜ₊₁ .*= c
        αₜ .= αₜ₊₁
    end
    check_nan(αₜ)
    return logL
end

# TODO: implement logdensityof(hmm, obs_seq)
