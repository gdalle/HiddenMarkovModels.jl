"""
    marginals!(γ, ξ, α, β, bβ, A)

Compute state and transition marginals by mutating `γ` and `ξ` (after forward and backward passes).
"""
function marginals!(γ::Matrix, ξ::Array, α::Matrix, β::Matrix, bβ::Matrix, A)
    T = size(γ, 2)
    @views for t in 1:T
        γ[:, t] .= α[:, t] .* β[:, t]
        γ_sum_inv = inv(sum(γ[:, t]))
        γ[:, t] .*= γ_sum_inv
    end
    check_nan(γ)
    @views for t in 1:(T - 1)
        ξ[:, :, t] .= α[:, t] .* A .* bβ[:, t + 1]'
        ξ_sum_inv = inv(sum(ξ[:, :, t]))
        ξ[:, :, t] .*= ξ_sum_inv
    end
    check_nan(ξ)
    return nothing
end
