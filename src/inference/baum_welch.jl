"""
$(TYPEDEF)

Store Baum-Welch quantities with element type `R` and observation type `O`.

Unlike the other storage types, this one is relative to multiple sequences.

# Fields

$(TYPEDFIELDS)
"""
struct BaumWelchStorage{R,O}
    "one `ForwardBackwardStorage` for each observation sequence"
    fbs::Vector{ForwardBackwardStorage{R}}
    "number of iterations performed"
    iteration::RefValue{Int}
    "history of total loglikelihood values throughout the algorithm"
    logL_evolution::Vector{R}
    "concatenation of `γ` matrices for all observation sequences (useful to avoid allocations in fitting)"
    state_marginals_concat::Matrix{R}
    "concatenation of observation sequences (useful to avoid allocations in fitting)"
    obs_seqs_concat::Vector{O}
    "temporal limits of each observation sequence in the concatenations"
    limits::Vector{Int}
end

function initialize_baum_welch(
    hmm::AbstractHMM, obs_seqs::Vector{<:Vector}, nb_seqs::Integer; max_iterations::Integer
)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    N, T = length(hmm), sum(length, obs_seqs)
    R = eltype(hmm, obs_seqs[1][1])
    fbs = Vector{ForwardBackwardStorage{R}}(undef, length(obs_seqs))
    @threads for k in eachindex(obs_seqs, fbs)
        fbs[k] = initialize_forward_backward(hmm, obs_seqs[k])
    end
    iteration = Ref(0)
    logL_evolution = Vector{R}(undef, max_iterations)
    state_marginals_concat = Matrix{R}(undef, N, T)
    obs_seqs_concat = reduce(vcat, obs_seqs)
    limits = vcat(0, cumsum(length.(obs_seqs)))
    return BaumWelchStorage(
        fbs, iteration, logL_evolution, state_marginals_concat, obs_seqs_concat, limits
    )
end

function baum_welch!(
    hmm::AbstractHMM,
    bw::BaumWelchStorage,
    obs_seqs::Vector{<:Vector};
    atol::Real,
    max_iterations::Integer,
    check_loglikelihood_increasing::Bool,
)
    @unpack (
        fbs, iteration, logL_evolution, state_marginals_concat, obs_seqs_concat, limits
    ) = bw
    iteration[] = 0

    while iteration[] < max_iterations
        # E step
        @threads for k in eachindex(obs_seqs, fbs)
            forward_backward!(fbs[k], hmm, obs_seqs[k])
            @views state_marginals_concat[:, (limits[k] + 1):limits[k + 1]] .= fbs[k].γ
        end

        # M step
        fit!(hmm, bw)

        # # Record likelihood
        iteration[] += 1
        logL_evolution[iteration[]] = sum(fb.logL[] for fb in fbs)

        # #  Stopping criterion
        if iteration[] > 1
            progress = logL_evolution[iteration[]] - logL_evolution[iteration[] - 1]
            if check_loglikelihood_increasing && progress < 0
                error("Loglikelihood decreased in Baum-Welch")
            elseif progress < atol
                break
            end
        end
    end
end

"""
    baum_welch(
        hmm_init, obs_seqs, nb_seqs;
        atol, max_iterations, check_loglikelihood_increasing
    )

Apply the Baum-Welch algorithm to estimate the parameters of an HMM starting from `hmm_init`, based on `nb_seqs` observation sequences.

Return a tuple `(hmm_est, logL_evolution)`.

!!! warning "Multithreading"
    This function is parallelized across sequences.

# Keyword arguments

- `atol`: Minimum loglikelihood increase at an iteration of the algorithm (otherwise the algorithm is deemed to have converged)
- `max_iterations`: Maximum number of iterations of the algorithm
- `check_loglikelihood_increasing`: Whether to throw an error if the loglikelihood decreases
"""
function baum_welch(
    hmm_init::AbstractHMM,
    obs_seqs::Vector{<:Vector},
    nb_seqs::Integer;
    atol=1e-5,
    max_iterations=100,
    check_loglikelihood_increasing=true,
)
    if nb_seqs != length(obs_seqs)
        throw(ArgumentError("nb_seqs != length(obs_seqs)"))
    end
    hmm = deepcopy(hmm_init)
    bw = initialize_baum_welch(hmm, obs_seqs, nb_seqs; max_iterations)
    baum_welch!(hmm, bw, obs_seqs; atol, max_iterations, check_loglikelihood_increasing)
    return hmm, bw.logL_evolution[1:bw.iteration[]]
end

"""
    baum_welch(
        hmm_init, obs_seq;
        atol, max_iterations, check_loglikelihood_increasing
    )

Apply the Baum-Welch algorithm to estimate the parameters of an HMM starting from `hmm_init`.

Return a tuple `(hmm_est, logL_evolution)`.

# Keyword arguments

- `atol`: Minimum loglikelihood increase at an iteration of the algorithm (otherwise the algorithm is deemed to have converged)
- `max_iterations`: Maximum number of iterations of the algorithm
- `check_loglikelihood_increasing`: Whether to throw an error if the loglikelihood decreases
"""
function baum_welch(
    hmm_init::AbstractHMM,
    obs_seq::Vector;
    atol=1e-5,
    max_iterations=100,
    check_loglikelihood_increasing=true,
)
    return baum_welch(
        hmm_init, [obs_seq], 1; atol, max_iterations, check_loglikelihood_increasing
    )
end
