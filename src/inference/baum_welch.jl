abstract type AbstractBaumWelchStorage end

"""
$(TYPEDEF)

# Fields

$(TYPEDFIELDS)
"""
struct BaumWelchStorage{O,R,M} <: AbstractBaumWelchStorage
    fb_storages::Vector{ForwardBackwardStorage{R,M}}
    obs_seqs_concat::Vector{O}
    state_marginals_concat::Matrix{R}
    seq_limits::Vector{Int}
    logL_evolution::Vector{R}
end

"""
    initialize_baum_welch(hmm, MultiSeq(obs_seqs); max_iterations)
"""
function initialize_baum_welch(hmm::AbstractHMM, obs_seqs::MultiSeq; max_iterations=0)
    O = typeof(obs_seqs[1][1])
    R = eltype(hmm, obs_seqs[1][1])
    fb_storages = initialize_forward_backward(hmm, obs_seqs)
    obs_seqs_concat = Vector{O}(undef, sum(length, obs_seqs))
    state_marginals_concat = Matrix{R}(undef, length(hmm), sum(length, obs_seqs))
    seq_limits = vcat(0, cumsum(length.(obs_seqs)))
    for k in eachindex(obs_seqs, fb_storages)
        obs_seqs_concat[(seq_limits[k] + 1):seq_limits[k + 1]] .= obs_seqs[k]
    end
    logL_evolution = R[]
    sizehint!(logL_evolution, max_iterations)
    bw_storage = BaumWelchStorage(
        fb_storages, obs_seqs_concat, state_marginals_concat, seq_limits, logL_evolution
    )
    return bw_storage
end

function update_baum_welch!(
    bw_storage::BaumWelchStorage, hmm::AbstractHMM, obs_seqs::MultiSeq
)
    @unpack fb_storages, state_marginals_concat, logL_evolution, seq_limits = bw_storage
    forward_backward!(fb_storages, hmm, obs_seqs)
    logL = zero(eltype(logL_evolution))
    for k in eachindex(obs_seqs, fb_storages)
        state_marginals_concat[:, (seq_limits[k] + 1):seq_limits[k + 1]] .= fb_storages[k].γ
        logL += fb_storages[k].logL[]
    end
    push!(logL_evolution, logL)
    return nothing
end

function baum_welch_has_converged(
    bw_storage::BaumWelchStorage; atol::Real, loglikelihood_increasing::Bool
)
    @unpack logL_evolution = bw_storage
    if length(logL_evolution) >= 2
        logL, logL_prev = logL_evolution[end], logL_evolution[end - 1]
        progress = logL - logL_prev
        if loglikelihood_increasing && progress < min(0, -atol)
            error("Loglikelihood decreased in Baum-Welch")
        elseif abs(progress) < atol
            return true
        end
    end
    return false
end

"""
    baum_welch!(
        bw_storage, hmm, MultiSeq(obs_seqs);
        atol, max_iterations, loglikelihood_increasing
    )
"""
function baum_welch!(
    bw_storage::AbstractBaumWelchStorage,
    hmm::AbstractHMM,
    obs_seqs::MultiSeq;
    atol::Real,
    max_iterations::Integer,
    loglikelihood_increasing::Bool,
)
    for _ in 1:max_iterations
        update_baum_welch!(bw_storage, hmm, obs_seqs)
        fit!(hmm, bw_storage, obs_seqs)
        if baum_welch_has_converged(bw_storage; atol, loglikelihood_increasing)
            break
        end
    end
    return nothing
end

"""
    baum_welch(hmm_guess, obs_seq; kwargs...)
    baum_welch(hmm_guess, MultiSeq(obs_seqs); kwargs...)

Apply the Baum-Welch algorithm to estimate the parameters of an HMM starting from `hmm_guess`.

Return a tuple `(hmm_est, logL_evolution)`.

# Keyword arguments

- `atol`: minimum loglikelihood increase at an iteration of the algorithm (otherwise the algorithm is deemed to have converged)
- `max_iterations`: maximum number of iterations of the algorithm
- `loglikelihood_increasing`: whether to throw an error if the loglikelihood decreases
"""
function baum_welch(
    hmm_guess::AbstractHMM,
    obs_seqs::MultiSeq;
    atol=1e-5,
    max_iterations=100,
    loglikelihood_increasing=true,
)
    hmm = deepcopy(hmm_guess)
    bw_storage = initialize_baum_welch(hmm, obs_seqs; max_iterations)
    baum_welch!(bw_storage, hmm, obs_seqs; atol, max_iterations, loglikelihood_increasing)
    return (; hmm, bw_storage.logL_evolution)
end

function baum_welch(hmm_guess::AbstractHMM, obs_seq::Vector; kwargs...)
    return baum_welch(hmm_guess, MultiSeq([obs_seq]); kwargs...)
end
