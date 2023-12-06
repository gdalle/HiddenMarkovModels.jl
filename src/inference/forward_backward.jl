"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct ForwardBackwardStorage{R,M<:AbstractMatrix{R}}
    "posterior state marginals `γ[i,t] = ℙ(X[t]=i | Y[1:T])`"
    γ::Matrix{R}
    "posterior transition marginals `ξ[t][i,j] = ℙ(X[t]=i, X[t+1]=j | Y[1:T])`"
    ξ::Vector{M}
    "loglikelihood of the observation sequence"
    logL::RefValue{R}
    α::Matrix{R}
    β::Matrix{R}
    c::Vector{R}
    logB::Matrix{R}
    logm::Vector{R}
    B::Matrix{R}
    scratch::Vector{R}
end

Base.eltype(::ForwardBackwardStorage{R}) where {R} = R

"""
$(SIGNATURES)
"""
function initialize_forward_backward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    transition_marginals=true,
)
    N, T = length(hmm), length(obs_seq)
    trans = transition_matrix(hmm, control_seq[1])
    R = eltype(hmm, obs_seq[1], control_seq[1])
    M = typeof(similar(trans, R))

    γ = Matrix{R}(undef, N, T)
    ξ = Vector{M}(undef, T)
    if transition_marginals
        for t in 1:T
            ξ[t] = similar(transition_matrix(hmm, control_seq[t]), R)
        end
    end
    logL = RefValue{R}()
    α = Matrix{R}(undef, N, T)
    β = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    logB = Matrix{R}(undef, N, T)
    logm = Vector{R}(undef, T)
    B = Matrix{R}(undef, N, T)
    scratch = Vector{R}(undef, N)
    return ForwardBackwardStorage{R,M}(γ, ξ, logL, α, β, c, logB, logm, B, scratch)
end

"""
$(SIGNATURES)
"""
function forward_backward!(
    storage::ForwardBackwardStorage{R},
    hmm::AbstractHMM,
    obs_seq::AbstractVector;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    transition_marginals::Bool=true,
) where {R}
    @unpack logL, α, β, c, γ, ξ, logB, logm, B, scratch = storage

    @views for t in eachindex(obs_seq, control_seq)
        obs_logdensities!(logB[:, t], hmm, obs_seq[t], control_seq[t])
    end
    maximum!(logm', logB)
    B .= exp.(logB .- logm')

    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)

        # Forward
        @views begin
            init = initialization(hmm)
            α[:, t1] .= init .* B[:, t1]
            c[t1] = inv(sum(α[:, t1]))
            lmul!(c[t1], α[:, t1])
        end
        @views for t in t1:(t2 - 1)
            trans = transition_matrix(hmm, control_seq[t])
            mul!(α[:, t + 1], trans', α[:, t])
            α[:, t + 1] .*= B[:, t + 1]
            c[t + 1] = inv(sum(α[:, t + 1]))
            lmul!(c[t + 1], α[:, t + 1])
        end

        # Backward and transition marginals
        β[:, t2] .= c[t2]
        if transition_marginals
            ξ[t2] .= zero(R)
        end
        @views for t in (t2 - 1):-1:t1
            trans = transition_matrix(hmm, control_seq[t])
            scratch .= B[:, t + 1] .* β[:, t + 1]  # Bβ
            mul!(β[:, t], trans, scratch)
            lmul!(c[t], β[:, t])
            if transition_marginals
                # transition marginals using Bβ
                mul_rows_cols!(ξ[t], view(α, :, t), trans, scratch)
            end
        end
    end

    # State marginals
    γ .= α .* β ./ c'
    check_finite(γ)

    # Loglikelihood
    logL[] = -sum(log, c) + sum(logm)
    return nothing
end

"""
$(SIGNATURES)

Apply the forward-backward algorithm to infer the posterior state and transition marginals during sequence `obs_seq` for `hmm`.

Return a tuple `(γ, logL)` defined in [`ForwardBackwardStorage`](@ref).

$(DESCRIBE_CONTROL_STARTS)
"""
function forward_backward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector;
    control_seq::AbstractVector=Fill(nothing, length(obs_seq)),
    seq_ends::AbstractVector{Int}=[length(obs_seq)],
)
    transition_marginals = false
    storage = initialize_forward_backward(
        hmm, obs_seq; control_seq, seq_ends, transition_marginals
    )
    forward_backward!(storage, hmm, obs_seq; control_seq, seq_ends, transition_marginals)
    return storage.γ, storage.logL[]
end
