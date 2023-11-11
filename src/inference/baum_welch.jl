"""
$(TYPEDEF)

Store Baum-Welch quantities with element type `R`.

This storage is relative to a several sequences.

# Fields

These fields (except `limits`) are passed to the `fit!(hmm, ...)` method along with `obs_seqs_concat`. 

$(TYPEDFIELDS)
"""
struct BaumWelchStorage{R,M<:AbstractMatrix{R}}
    "posterior initialization counts for each state"
    init_count::Vector{R}
    "posterior transition counts for each state"
    trans_count::M
    "concatenation along time of the state marginal matrices `γ[i,t] = ℙ(X[t]=i | Y[1:T])` for all observation sequences"
    state_marginals_concat::Matrix{R}
    "temporal separations between observation sequences: `state_marginals_concat[limits[k]+1:limits[k+1]]` refers to sequence `k`"
    limits::Vector{Int}
end

function initialize_baum_welch(
    hmm::AbstractHMM, obs_seqs::Vector{<:Vector}, nb_seqs::Integer
)
    check_nb_seqs(obs_seqs, nb_seqs)
    N, T_concat = length(hmm), sum(length, obs_seqs)
    A = transition_matrix(hmm)
    R = eltype(hmm, obs_seqs[1][1])
    init_count = Vector{R}(undef, N)
    trans_count = similar(A, R)
    state_marginals_concat = Matrix{R}(undef, N, T_concat)
    limits = vcat(0, cumsum(length.(obs_seqs)))
    return BaumWelchStorage(init_count, trans_count, state_marginals_concat, limits)
end

function initialize_logL_evolution(
    hmm::AbstractHMM, obs_seqs::Vector{<:Vector}, nb_seqs::Integer; max_iterations::Integer
)
    check_nb_seqs(obs_seqs, nb_seqs)
    R = eltype(hmm, obs_seqs[1][1])
    logL_evolution = R[]
    sizehint!(logL_evolution, max_iterations)
    return logL_evolution
end

function update_sufficient_statistics!(
    bw::BaumWelchStorage{R}, fbs::Vector{<:ForwardBackwardStorage}
) where {R}
    @unpack init_count, trans_count, state_marginals_concat, limits = bw
    init_count .= zero(R)
    trans_count .= zero(R)
    state_marginals_concat .= zero(R)
    for k in eachindex(fbs)  # TODO: ThreadsX?
        init_count .+= fbs[k].init_count
        mynonzeros(trans_count) .+= mynonzeros(fbs[k].trans_count)
        state_marginals_concat[:, (limits[k] + 1):limits[k + 1]] .= fbs[k].γ
    end
    return nothing
end

function baum_welch_has_converged(
    logL_evolution::Vector; atol::Real, loglikelihood_increasing::Bool
)
    if length(logL_evolution) >= 2
        logL, logL_prev = logL_evolution[end], logL_evolution[end - 1]
        progress = logL - logL_prev
        if loglikelihood_increasing && progress < 0
            error("Loglikelihood decreased in Baum-Welch")
        elseif progress < atol
            return true
        end
    end
    return false
end

function StatsAPI.fit!(hmm::AbstractHMM, bw::BaumWelchStorage, obs_seqs_concat::Vector)
    return fit!(
        hmm, bw.init_count, bw.trans_count, obs_seqs_concat, bw.state_marginals_concat
    )
end

function baum_welch!(
    fbs::Vector{<:ForwardBackwardStorage},
    bw::BaumWelchStorage,
    logL_evolution::Vector,
    hmm::AbstractHMM,
    obs_seqs::Vector{<:Vector},
    obs_seqs_concat::Vector;
    atol::Real,
    max_iterations::Integer,
    loglikelihood_increasing::Bool,
)
    for _ in 1:max_iterations
        @threads for k in eachindex(obs_seqs, fbs)
            forward_backward!(fbs[k], hmm, obs_seqs[k])
        end
        update_sufficient_statistics!(bw, fbs)
        push!(logL_evolution, sum(fb.logL[] for fb in fbs))
        fit!(hmm, bw, obs_seqs_concat)
        check_hmm(hmm)
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
    fbs = [initialize_forward_backward(hmm, obs_seqs[k]) for k in eachindex(obs_seqs)]
    bw = initialize_baum_welch(hmm, obs_seqs, nb_seqs)
    logL_evolution = initialize_logL_evolution(hmm, obs_seqs, nb_seqs; max_iterations)
    obs_seqs_concat = reduce(vcat, obs_seqs)
    baum_welch!(
        fbs,
        bw,
        logL_evolution,
        hmm,
        obs_seqs,
        obs_seqs_concat;
        atol,
        max_iterations,
        loglikelihood_increasing,
    )
    return hmm, logL_evolution
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
