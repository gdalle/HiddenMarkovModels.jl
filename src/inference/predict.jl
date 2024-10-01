function predict_next_state!(
    next_state_marginals::AbstractVector{<:Real},
    hmm::AbstractHMM,
    current_state_marginals::AbstractVector{<:Real},
    control=nothing,
)
    trans = transition_matrix(hmm, control)
    mul!(next_state_marginals, transpose(trans), current_state_marginals)
    return next_state_marginals
end

function predict_next_state(
    hmm::AbstractHMM, current_state_marginals::AbstractVector{<:Real}, control=nothing
)
    next_state_marginals = similar(current_state_marginals)
    return predict_next_state!(next_state_marginals, hmm, current_state_marginals, control)
end
