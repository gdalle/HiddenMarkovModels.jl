abstract type AbstractBaumWelchStorage end

"""
    initialize_logL_evolution(hmm, obs_seqs, nb_seqs; max_iterations)
"""
function initialize_logL_evolution(
    hmm::AbstractHMM, obs_seqs::Vector{<:Vector}, nb_seqs::Integer; max_iterations::Integer
)
    check_nb_seqs(obs_seqs, nb_seqs)
    R = eltype(hmm, obs_seqs[1][1])
    logL_evolution = R[]
    sizehint!(logL_evolution, max_iterations)
    return logL_evolution
end

function baum_welch_has_converged(
    logL_evolution::Vector; atol::Real, loglikelihood_increasing::Bool
)
    if length(logL_evolution) >= 2
        logL, logL_prev = logL_evolution[end], logL_evolution[end - 1]
        progress = logL - logL_prev
        if loglikelihood_increasing && progress < 0
            error("Loglikelihood decreased in Baum-Welch")
        elseif abs(progress) < atol
            return true
        end
    end
    return false
end

"""
    baum_welch!(
        fb_storages, bw_storage, logL_evolution,
        hmm, obs_seqs;
        atol, max_iterations, loglikelihood_increasing
    )
"""
function baum_welch!(
    hmm::AbstractHMM,
    fb_storages::Vector{<:ForwardBackwardStorage},
    bw_storage::AbstractBaumWelchStorage,
    logL_evolution::Vector,
    obs_seqs::Vector{<:Vector};
    atol::Real,
    max_iterations::Integer,
    loglikelihood_increasing::Bool,
)
    for _ in 1:max_iterations
        for k in eachindex(obs_seqs, fb_storages)
            forward_backward!(fb_storages[k], hmm, obs_seqs[k])
        end
        push!(logL_evolution, sum(fb.logL[] for fb in fb_storages))
        fit!(hmm, bw_storage, fb_storages, obs_seqs)
        if baum_welch_has_converged(logL_evolution; atol, loglikelihood_increasing)
            break
        end
    end
    return nothing
end

"""
    baum_welch(hmm_init, obs_seq; kwargs...)
    baum_welch(hmm_init, obs_seqs, nb_seqs; kwargs...)

Apply the Baum-Welch algorithm to estimate the parameters of an HMM starting from `hmm_init`.

Return a tuple `(hmm_est, logL_evolution)`.

# Keyword arguments

- `atol`: minimum loglikelihood increase at an iteration of the algorithm (otherwise the algorithm is deemed to have converged)
- `max_iterations`: maximum number of iterations of the algorithm
- `loglikelihood_increasing`: whether to throw an error if the loglikelihood decreases
"""
function baum_welch(
    hmm_init::AbstractHMM,
    obs_seqs::Vector{<:Vector},
    nb_seqs::Integer;
    atol=1e-5,
    max_iterations=100,
    loglikelihood_increasing=true,
)
    check_nb_seqs(obs_seqs, nb_seqs)
    hmm = deepcopy(hmm_init)
    fb_storages = [
        initialize_forward_backward(hmm, obs_seqs[k]) for k in eachindex(obs_seqs)
    ]
    bw_storage = initialize_baum_welch(hmm, fb_storages, obs_seqs)
    logL_evolution = initialize_logL_evolution(hmm, obs_seqs, nb_seqs; max_iterations)
    baum_welch!(
        hmm,
        fb_storages,
        bw_storage,
        logL_evolution,
        obs_seqs;
        atol,
        max_iterations,
        loglikelihood_increasing,
    )
    return (; hmm, logL_evolution)
end

function baum_welch(
    hmm_init::AbstractHMM,
    obs_seq::Vector;
    atol=1e-5,
    max_iterations=100,
    loglikelihood_increasing=true,
)
    return baum_welch(
        hmm_init, [obs_seq], 1; atol, max_iterations, loglikelihood_increasing
    )
end
