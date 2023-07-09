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
    check_no_nan(α)
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
    check_no_nan(β)
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
    check_no_nan(γ)
    @views for t in 1:(T - 1)
        ξ[:, :, t] .= α[:, t] .* A .* _Bβ[:, t + 1]'
        normalization = inv(sum(ξ[:, :, t]))
        ξ[:, :, t] .*= normalization
    end
    check_no_nan(ξ)
    return nothing
end

function forward_backward!(fb::ForwardBackwardStorage, hmm::AbstractHMM, logB)
    p = initial_distribution(hmm)
    A = transition_matrix(hmm)
    forward!(fb, p, A, logB)
    backward!(fb, A, logB)
    marginals!(fb, A)
    return nothing
end

function forward_backward_from_loglikelihoods(hmm::AbstractHMM, logB)
    fb = initialize_forward_backward(hmm, logB)
    forward_backward!(fb, hmm, logB)
    return fb
end

"""
    forward_backward(hmm, obs_seq)

Apply the forward-backward algorithm to estimate the posterior state marginals of an HMM for a single observation sequence, and return a [`ForwardBackwardStorage`](@ref).
"""
function forward_backward(hmm::AbstractHMM, obs_seq)
    logB = loglikelihoods(hmm, obs_seq)
    return forward_backward_from_loglikelihoods(hmm, logB)
end

"""
    forward_backward(hmm, obs_seqs, nb_seqs)

Apply the forward-backward algorithm to estimate the posterior state marginals of an HMM for multiple observation sequences, and return a [`ForwardBackwardStorage`](@ref).
"""
function forward_backward(hmm::AbstractHMM, obs_seqs, nb_seqs::Integer)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    fb1 = forward_backward(hmm, first(obs_seqs))
    fbs = Vector{typeof(fb1)}(undef, nb_seqs)
    fbs[1] = fb1
    @threads for k in 2:nb_seqs
        fbs[k] = forward_backward(hmm, obs_seqs[k])
    end
    return fbs
end
