"""
$(TYPEDEF)

Basic implementation of an HMM.

# Fields

$(TYPEDFIELDS)
"""
struct HMM{
    V<:AbstractVector,
    M<:AbstractMatrix,
    VD<:AbstractVector,
    Vl<:AbstractVector,
    Ml<:AbstractMatrix,
    Mt<:AbstractMatrix,
    Mlt<:AbstractMatrix,
} <: AbstractHMM
    "initial state probabilities"
    init::V
    "state transition probabilities"
    trans::M
    "observation distributions"
    dists::VD
    "logarithms of initial state probabilities"
    loginit::Vl
    "logarithms of state transition probabilities"
    logtrans::Ml
    transpose_trans::Mt
    transpose_logtrans::Mlt

    function HMM(init::AbstractVector, trans::AbstractMatrix, dists::AbstractVector)
        loginit = elementwise_log(init)
        logtrans = elementwise_log(trans)
        transpose_trans = concrete_transpose(trans)
        transpose_logtrans = concrete_transpose(logtrans)
        hmm = new{
            typeof(init),
            typeof(trans),
            typeof(dists),
            typeof(loginit),
            typeof(logtrans),
            typeof(transpose_trans),
            typeof(transpose_logtrans),
        }(
            init, trans, dists, loginit, logtrans, transpose_trans, transpose_logtrans
        )
        @argcheck valid_hmm(hmm)
        return hmm
    end
end

function Base.copy(hmm::HMM)
    return HMM(copy(hmm.init), copy(hmm.trans), copy(hmm.dists))
end

function Base.show(io::IO, hmm::HMM)
    return print(
        io,
        "Hidden Markov Model with:\n - initialization: $(hmm.init)\n - transition matrix: $(hmm.trans)\n - observation distributions: [$(join(hmm.dists, ", "))]",
    )
end

initialization(hmm::HMM) = hmm.init
log_initialization(hmm::HMM) = hmm.loginit
transition_matrix(hmm::HMM) = hmm.trans
transpose_transition_matrix(hmm::HMM) = hmm.transpose_trans
log_transition_matrix(hmm::HMM) = hmm.logtrans
transpose_log_transition_matrix(hmm::HMM) = hmm.transpose_logtrans
obs_distributions(hmm::HMM) = hmm.dists

## Fitting

function StatsAPI.fit!(
    hmm::HMM,
    fb_storage::ForwardBackwardStorage,
    obs_seq::AbstractVector;
    seq_ends::AbstractVector{Int},
)
    (; γ, ξ) = fb_storage
    # Fit states
    @threads for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        # use ξ[t2] as scratch space since it is zero anyway
        scratch = ξ[t2]
        scratch .= zero(eltype(scratch))
        for t in t1:(t2 - 1)
            scratch .+= ξ[t]
        end
    end
    hmm.init .= zero(eltype(hmm.init))
    hmm.trans .= zero(eltype(hmm.trans))
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        hmm.init .+= view(γ, :, t1)
        hmm.trans .+= ξ[t2]
    end
    sum_to_one!(hmm.init)
    foreach(sum_to_one!, eachrow(hmm.trans))
    # Fit observations
    for i in 1:length(hmm)
        fit_in_sequence!(hmm.dists, i, obs_seq, view(γ, i, :))
    end
    # Update logs
    hmm.loginit .= log.(hmm.init)
    mynonzeros(hmm.logtrans) .= log.(mynonzeros(hmm.trans))
    # Update transposes (could be optimized)
    copyto!(hmm.transpose_trans, transpose(hmm.trans))
    copyto!(hmm.transpose_logtrans, transpose(hmm.logtrans))
    # Safety check
    @argcheck valid_hmm(hmm)
    return nothing
end
