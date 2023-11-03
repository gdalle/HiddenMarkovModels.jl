"""
$(TYPEDEF)

Store Baum-Welch quantities with element type `R` and observation type `O`.

# Fields

$(TYPEDFIELDS)
"""
struct BaumWelchStorage{R,O}
    fbs::Vector{ForwardBackwardStorage{R}}
    logL_evolution::Vector{R}
    state_marginals_concat::Matrix{R}
    obs_seqs_concat::Vector{O}
    limits::Vector{Int}
end

function initialize_baum_welch(hmm::AbstractHMM, obs_seqs::Vector{<:Vector}; max_iterations)
    N, T = length(hmm), sum(length, obs_seqs)
    R = eltype(hmm, obs_seqs[1][1])
    fbs = Vector{ForwardBackwardStorage{R}}(undef, length(obs_seqs))
    @threads for k in eachindex(obs_seqs, fbs)
        fbs[k] = initialize_forward_backward(hmm, obs_seqs[k])
    end
    logL_evolution = Vector{R}(undef, max_iterations)
    state_marginals_concat = Matrix{R}(undef, N, T)
    obs_seqs_concat = reduce(vcat, obs_seqs)
    limits = vcat(0, cumsum(length.(obs_seqs)))
    return BaumWelchStorage(
        fbs, logL_evolution, state_marginals_concat, obs_seqs_concat, limits
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
    @unpack fbs, logL_evolution, state_marginals_concat, obs_seqs_concat, limits = bw

    iteration = 0
    while iteration < max_iterations
        # E step
        @threads for k in eachindex(obs_seqs, fbs)
            forward_backward!(fbs[k], hmm, obs_seqs[k])
            @views state_marginals_concat[:, (limits[k] + 1):limits[k + 1]] .= fbs[k].γ
        end
        logL_evolution[iteration] = sum(fb.logL[] for fb in fbs)

        # M step
        fit!(hmm, bw, obs_seqs)

        #  Stopping criterion
        iteration += 1
        if iteration > 1
            progress = logL_evolution[iteration] - logL_evolution[iteration - 1]
            if check_loglikelihood_increasing && progress < 0
                error("Loglikelihood decreased in Baum-Welch")
            elseif progress < atol
                break
            end
        end
    end

    logL_evolution = logL_evolution[1:iteration]
    return nothing
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
    bw = initialize_baum_welch(hmm, obs_seqs; max_iterations)
    baum_welch!(hmm, bw, obs_seqs; atol, max_iterations, check_loglikelihood_increasing)
    return hmm, bw.logL_evolution
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
    obs_seq;
    atol=1e-5,
    max_iterations=100,
    check_loglikelihood_increasing=true,
)
    return baum_welch(
        hmm_init, [obs_seq]; atol, max_iterations, check_loglikelihood_increasing
    )
end
