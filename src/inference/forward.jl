"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct ForwardStorage{R}
    "posterior last state marginals `α[i] = ℙ(X[T]=i | Y[1:T])`"
    α::Matrix{R}
    "one loglikelihood per observation sequence"
    logL::Vector{R}
    B::Matrix{R}
    c::Vector{R}
end

Base.eltype(::ForwardStorage{R}) where {R} = R

"""
$(SIGNATURES)
"""
function initialize_forward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
)
    N, T, K = length(hmm), length(obs_seq), length(seq_ends)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    α = Matrix{R}(undef, N, T)
    logL = Vector{R}(undef, K)
    B = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    return ForwardStorage(α, logL, B, c)
end

"""
$(SIGNATURES)
"""
function forward!(
    storage,
    hmm::AbstractHMM,
    obs_seq::AbstractVector;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
)
    @unpack α, logL, B, c = storage

    @views for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)

        # Initialization
        Bₜ₁ = B[:, t1]
        obs_logdensities!(Bₜ₁, hmm, obs_seq[t1], control_seq[t1])
        logm = maximum(Bₜ₁)
        Bₜ₁ .= exp.(Bₜ₁ .- logm)

        init = initialization(hmm)
        αₜ₁ = α[:, t1]
        αₜ₁ .= init .* Bₜ₁
        c[t1] = inv(sum(αₜ₁))
        lmul!(c[t1], αₜ₁)

        logL[k] = -log(c[t1]) + logm

        # Loop
        for t in t1:(t2 - 1)
            Bₜ₊₁ = B[:, t + 1]
            obs_logdensities!(Bₜ₊₁, hmm, obs_seq[t + 1], control_seq[t + 1])
            logm = maximum(Bₜ₊₁)
            Bₜ₊₁ .= exp.(Bₜ₊₁ .- logm)

            trans = transition_matrix(hmm, control_seq[t])
            αₜ₊₁ = α[:, t + 1]
            mul!(αₜ₊₁, trans', α[:, t])
            αₜ₊₁ .*= Bₜ₊₁
            c[t + 1] = inv(sum(αₜ₊₁))
            lmul!(c[t + 1], αₜ₊₁)

            logL[k] += -log(c[t + 1]) + logm
        end
    end

    check_finite(α)
    return nothing
end

"""
$(SIGNATURES)

Apply the forward algorithm to infer the current state after sequence `obs_seq` for `hmm`.
    
Return a tuple `(storage.α, sum(storage.logL))` where `storage` is of type [`ForwardStorage`](@ref).
"""
function forward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector;
    control_seq::AbstractVector=Fill(nothing, length(obs_seq)),
    seq_ends::AbstractVector{Int}=Fill(length(obs_seq), 1),
)
    storage = initialize_forward(hmm, obs_seq; control_seq, seq_ends)
    forward!(storage, hmm, obs_seq; control_seq, seq_ends)
    return storage.α, sum(storage.logL)
end
