function forward!(fb::ForwardBackwardStorage, p, A, logB)
    (; α, _c, _m) = fb
    T = size(α, 2)
    @views begin
        _m[1] = maximum(logB[:, 1])
        α[:, 1] .= p .* exp.(logB[:, 1] .- _m[1])
        _c[1] = inv(sum(α[:, 1]))
        α[:, 1] .*= _c[1]
    end
    @views for t in 1:(T - 1)
        _m[t + 1] = maximum(logB[:, t + 1])
        mul!(α[:, t + 1], A', α[:, t])
        α[:, t + 1] .*= exp.(logB[:, t + 1] .- _m[t + 1])
        _c[t + 1] = inv(sum(α[:, t + 1]))
        α[:, t + 1] .*= _c[t + 1]
    end
    return nothing
end

function backward!(fb::ForwardBackwardStorage{R}, A, logB) where {R}
    (; β, _c, _m, _Bβ) = fb
    T = size(β, 2)
    β[:, T] .= one(R)
    @views for t in (T - 1):-1:1
        _Bβ[:, t + 1] .= exp.(logB[:, t + 1] .- _m[t + 1]) .* β[:, t + 1]
        mul!(β[:, t], A, _Bβ[:, t + 1])
        β[:, t] .*= _c[t]
    end
    return nothing
end

function marginals!(fb::ForwardBackwardStorage, A)
    (; α, β, _Bβ, γ, ξ) = fb
    T = size(γ, 2)
    @views for t in 1:T
        γ[:, t] .= α[:, t] .* β[:, t]
        normalization = inv(sum(γ[:, t]))
        γ[:, t] .*= normalization
    end
    @views for t in 1:(T - 1)
        ξ[:, :, t] .= α[:, t] .* A .* _Bβ[:, t + 1]'
        normalization = inv(sum(ξ[:, :, t]))
        ξ[:, :, t] .*= normalization
    end
    return nothing
end

function forward_backward!(fb::ForwardBackwardStorage, sp::StateProcess, logB)
    p = initial_distribution(sp)
    A = transition_matrix(sp)
    forward!(fb, p, A, logB)
    backward!(fb, A, logB)
    marginals!(fb, A)
    return nothing
end

"""
    forward_backward(hmm, obs_seq)

Apply the forward-backward algorithm to estimate the posterior state marginals of an HMM.
"""
function forward_backward(hmm::HMM, obs_seq)
    logB = loglikelihoods(hmm.obs_process, obs_seq)
    fb = initialize_forward_backward(hmm.state_process, logB)
    forward_backward!(fb, hmm.state_process, logB)
    return fb
end
