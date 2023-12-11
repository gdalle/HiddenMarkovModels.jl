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
    "one loglikelihood per observation sequence"
    logL::Vector{R}
    B::Matrix{R}
    α::Matrix{R}
    c::Vector{R}
    β::Matrix{R}
    Bβ::Matrix{R}
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
    N, T, K = length(hmm), length(obs_seq), length(seq_ends)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    trans = transition_matrix(hmm, control_seq[1])
    M = typeof(mysimilar_mutable(trans, R))

    γ = Matrix{R}(undef, N, T)
    ξ = Vector{M}(undef, T)
    if transition_marginals
        for t in 1:T
            ξ[t] = mysimilar_mutable(transition_matrix(hmm, control_seq[t]), R)
        end
    end
    logL = Vector{R}(undef, K)
    B = Matrix{R}(undef, N, T)
    α = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    β = Matrix{R}(undef, N, T)
    Bβ = Matrix{R}(undef, N, T)
    return ForwardBackwardStorage{R,M}(γ, ξ, logL, B, α, c, β, Bβ)
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
    @unpack logL, α, β, c, γ, ξ, B, Bβ = storage

    # Forward (fill B, α, c and logL)
    forward!(storage, hmm, obs_seq; control_seq, seq_ends)

    @views for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)

        # Backward
        β[:, t2] .= c[t2]
        for t in (t2 - 1):-1:t1
            trans = transition_matrix(hmm, control_seq[t])
            Bβ[:, t + 1] .= B[:, t + 1] .* β[:, t + 1]
            mul!(β[:, t], trans, Bβ[:, t + 1])
            lmul!(c[t], β[:, t])
        end
        Bβ[:, t1] .= B[:, t1] .* β[:, t1]

        # State marginals
        γ[:, t1:t2] .= α[:, t1:t2] .* β[:, t1:t2] ./ c[t1:t2]'

        # Transition marginals
        if transition_marginals
            for t in t1:(t2 - 1)
                trans = transition_matrix(hmm, control_seq[t])
                mul_rows_cols!(ξ[t], α[:, t], trans, Bβ[:, t + 1])
            end
            ξ[t2] .= zero(R)
        end
    end

    check_finite(γ)
    return nothing
end

"""
$(SIGNATURES)

Apply the forward-backward algorithm to infer the posterior state and transition marginals during sequence `obs_seq` for `hmm`.

Return a tuple `(storage.γ, sum(storage.logL))` where `storage` is of type [`ForwardBackwardStorage`](@ref).

$(DESCRIBE_CONTROL_STARTS)
"""
function forward_backward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector;
    control_seq::AbstractVector=Fill(nothing, length(obs_seq)),
    seq_ends::AbstractVector{Int}=Fill(length(obs_seq), 1),
)
    transition_marginals = false
    storage = initialize_forward_backward(
        hmm, obs_seq; control_seq, seq_ends, transition_marginals
    )
    forward_backward!(storage, hmm, obs_seq; control_seq, seq_ends, transition_marginals)
    return storage.γ, sum(storage.logL)
end
