"""
$(SIGNATURES)
"""
function initialize_forward_backward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    transition_marginals=true,
)
    N, T, K = length(hmm), length(obs_seq), length(seq_ends)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    trans = transition_matrix(hmm, control_seq[1])
    M = typeof(similar(trans, R))

    γ = Matrix{R}(undef, N, T)
    ξ = Vector{M}(undef, T)
    if transition_marginals
        for t in 1:T
            ξ[t] = similar(transition_matrix(hmm, control_seq[t]), R)
        end
    end
    logL = Vector{R}(undef, K)
    logB = Matrix{R}(undef, N, T)
    α = Matrix{R}(undef, N, T)
    logα = Matrix{R}(undef, N, T)
    logβ = Matrix{R}(undef, N, T)
    logγ = Matrix{R}(undef, N, T)
    logξ = Vector{M}(undef, T)
    if transition_marginals
        for t in 1:T
            logξ[t] = similar(log_transition_matrix(hmm, control_seq[t]), R)
        end
    end
    tmp = Vector{R}(undef, N)
    return ForwardBackwardStorage{R,M}(γ, ξ, logL, logB, α, logα, logβ, logγ, logξ, tmp)
end

function _forward_backward!(
    storage::ForwardBackwardStorage{R},
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer;
    transition_marginals::Bool=true,
) where {R}
    (; logα, logβ, logγ, logξ, γ, ξ, logB, tmp) = storage
    t1, t2 = seq_limits(seq_ends, k)

    # Forward (fill B, logα, and logL)
    _forward!(storage, hmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite=true)

    # Backward
    logβ[:, t2] .= zero(R)
    for t in (t2 - 1):-1:t1
        logβₜ = view(logβ, :, t)
        logβₜ₊₁ = view(logβ, :, t + 1)
        logBₜ₊₁ = view(logB, :, t + 1)
        logtrans = log_transition_matrix(hmm, control_seq[t])
        for i in eachindex(logβₜ)
            tmp .= view(logtrans, i, :) .+ logBₜ₊₁ .+ logβₜ₊₁
            logβₜ[i] = logsumexp(tmp)
        end
    end

    # State marginals
    for t in t1:t2
        logγ[:, t] .= view(logα, :, t) .+ view(logβ, :, t)
        logγ[:, t] .-= logsumexp(view(logγ, :, t))
        γ[:, t] .= exp.(view(logγ, :, t))
    end

    # Transition marginals
    if transition_marginals
        for t in t1:(t2 - 1)
            logtrans = log_transition_matrix(hmm, control_seq[t])
            tmp .= view(logB, :, t + 1) .+ view(logβ, :, t + 1)
            add_rows_cols!(logξ[t], view(logα, :, t), logtrans, tmp)
            logξ[t] .-= logsumexp(logξ[t])
            ξ[t] .= exp.(logξ[t])
        end
        logξ[t2] .= log(zero(R))
        ξ[t2] .= zero(R)
    end

    return nothing
end

"""
$(SIGNATURES)
"""
function forward_backward!(
    storage::ForwardBackwardStorage,
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    transition_marginals::Bool=true,
)
    if seq_ends isa NTuple{1}
        for k in eachindex(seq_ends)
            _forward_backward!(
                storage, hmm, obs_seq, control_seq, seq_ends, k; transition_marginals
            )
        end
    else
        @threads for k in eachindex(seq_ends)
            _forward_backward!(
                storage, hmm, obs_seq, control_seq, seq_ends, k; transition_marginals
            )
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the forward-backward algorithm to infer the posterior state and transition marginals during sequence `obs_seq` for `hmm`.

Return a tuple `(storage.γ, storage.logL)` where `storage` is of type [`ForwardBackwardStorage`](@ref).
"""
function forward_backward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    transition_marginals = false
    storage = initialize_forward_backward(
        hmm, obs_seq, control_seq; seq_ends, transition_marginals
    )
    forward_backward!(storage, hmm, obs_seq, control_seq; seq_ends, transition_marginals)
    return storage.γ, storage.logL
end
