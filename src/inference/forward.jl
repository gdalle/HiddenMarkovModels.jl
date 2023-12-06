"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct ForwardStorage{R}
    "posterior last state marginals `α[i] = ℙ(X[T]=i | Y[1:T])"
    α::Vector{R}
    "loglikelihood of the observation sequence"
    logL::RefValue{R}
    logb::Vector{R}
    α_next::Vector{R}
end

Base.eltype(::ForwardStorage{R}) where {R} = R

"""
    initialize_forward(hmm, obs_seq; control_seq, seq_ends)
"""
function initialize_forward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
)
    N = length(hmm)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    α = Vector{R}(undef, N)
    logL = RefValue{R}()
    logb = Vector{R}(undef, N)
    α_next = Vector{R}(undef, N)
    storage = ForwardStorage(α, logL, logb, α_next)
    return storage
end

"""
    forward!(storage, hmm, obs_seq; control_seq, seq_ends)
"""
function forward!(
    storage::ForwardStorage{R},
    hmm::AbstractHMM,
    obs_seq::AbstractVector;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
) where {R}
    @unpack logL, logb, α, α_next = storage
    logL[] = zero(R)
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        init = initialization(hmm)
        obs_logdensities!(logb, hmm, obs_seq[t1], control_seq[t1])
        logm = maximum(logb)
        α .= init .* exp.(logb .- logm)
        c = inv(sum(α))
        α .*= c
        logL[] += -log(c) + logm
        for t in t1:(t2 - 1)
            trans = transition_matrix(hmm, control_seq[t])
            obs_logdensities!(logb, hmm, obs_seq[t + 1], control_seq[t + 1])
            logm = maximum(logb)
            mul!(α_next, trans', α)
            α_next .*= exp.(logb .- logm)
            c = inv(sum(α_next))
            α_next .*= c
            α .= α_next
            logL[] += -log(c) + logm
        end
        check_finite(α)
    end
    return nothing
end

"""
    forward(hmm, obs_seq; control_seq, seq_ends)

Apply the forward algorithm to infer the current state after sequence `obs_seq` for `hmm`.
    
Return a tuple `(α, logL)` defined in [`ForwardStorage`](@ref).

# Keyword arguments

$(DESCRIBE_CONTROL_STARTS)
"""
function forward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector;
    control_seq::AbstractVector=Fill(nothing, length(obs_seq)),
    seq_ends::AbstractVector{Int}=[length(obs_seq)],
)
    storage = initialize_forward(hmm, obs_seq; control_seq, seq_ends)
    forward!(storage, hmm, obs_seq; control_seq, seq_ends)
    return storage.α, storage.logL[]
end
